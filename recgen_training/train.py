import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict
import warnings

# Suppress common warnings BEFORE importing torch
warnings.filterwarnings('ignore', message='The pynvml package is deprecated')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda')

import torch
import torch.multiprocessing as mp
import numpy as np
import random

from recgen_inference.recgen_modules import models
from recgen_training import datasets, trainers
from recgen_training.utils.dist_utils import setup_dist, master_first


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def expand_path_placeholders(obj, data_dir, config_dir):
    """
    Recursively expand {data_dir} and {config_dir} placeholders in all string
    values of a (nested) config structure. This keeps the shipped configs
    path-portable: shard patterns and stats files reference the user's data
    root and the config's own location instead of hardcoded absolute paths.
    """
    if isinstance(obj, str):
        if '{data_dir}' in obj or '{config_dir}' in obj:
            return obj.replace('{data_dir}', data_dir).replace('{config_dir}', config_dir)
        return obj
    if isinstance(obj, dict):
        return type(obj)({k: expand_path_placeholders(v, data_dir, config_dir) for k, v in obj.items()})
    if isinstance(obj, list):
        return [expand_path_placeholders(v, data_dir, config_dir) for v in obj]
    return obj


def warn_attention_backend():
    """
    The released checkpoints were trained with ATTN_BACKEND=flash_attn.
    Other backends (xformers/sdpa) are functional but do not bit-reproduce
    the original training numerics.
    """
    from recgen_inference.recgen_modules.modules import attention, sparse
    if attention.BACKEND != 'flash_attn' or sparse.ATTN != 'flash_attn':
        print(
            '\n\033[93m'
            'WARNING: attention backend is '
            f'(dense={attention.BACKEND}, sparse={sparse.ATTN}), but the paper '
            'checkpoints were trained with flash_attn. Training still works, '
            'but results will not bit-reproduce the original runs. Install '
            'flash-attn and/or set ATTN_BACKEND=flash_attn to match.'
            '\033[0m\n'
        )


def load_dataset_from_config(cfg, dataset_key='dataset', default_name=None):
    """
    Load a dataset from config.

    Args:
        cfg: Config object
        dataset_key: Key in config for the dataset (e.g., 'dataset', 'val_dataset', 'train_val_dataset')
        default_name: Default dataset name if not specified in the dataset config

    Returns:
        Dataset object or None if not configured
    """
    if not hasattr(cfg, dataset_key):
        return None

    dataset_cfg = getattr(cfg, dataset_key)
    if not dataset_cfg:
        return None

    dataset_args = dataset_cfg.args.copy()
    dataset_name = dataset_cfg.name if hasattr(dataset_cfg, 'name') else (default_name or cfg.dataset.name)

    # Auto-detect if pointmaps are required based on model config
    # Check if model uses point_embedder
    if dataset_name in ('WebDatasetImageConditionedSparseStructureLatent', 'WebDatasetMultiViewSparseStructureLatent'):
        if 'require_pointmap' not in dataset_args:
            # Check if any model in cfg.models has use_point_embedder=True
            requires_pointmap = False
            if hasattr(cfg, 'models'):
                for model_name, model_cfg in cfg.models.items():
                    if hasattr(model_cfg, 'args') and model_cfg.args.get('use_point_embedder', False):
                        requires_pointmap = True
                        break
            if requires_pointmap:
                dataset_args['require_pointmap'] = True

    # If 'roots' is already in dataset_args, don't pass cfg.data_dir
    # Otherwise, pass cfg.data_dir as the first positional argument (roots)
    if 'roots' in dataset_args:
        return getattr(datasets, dataset_name)(**dataset_args)
    else:
        return getattr(datasets, dataset_name)(cfg.data_dir, **dataset_args)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    if rank == 0:
        warn_attention_backend()

    # Load datasets from config
    # Use master_first to ensure rank 0 computes and caches pose stats before other ranks
    # This prevents race conditions when multiple processes try to compute stats simultaneously
    with master_first():
        dataset = load_dataset_from_config(cfg, 'dataset')
        val_dataset = load_dataset_from_config(cfg, 'val_dataset')
        # subset of train for validation metric computation
        train_val_dataset = load_dataset_from_config(cfg, 'train_val_dataset')

    if rank == 0:
        print(f'Loaded training dataset: {len(dataset)} samples')
        if val_dataset:
            print(f'Loaded validation dataset: {len(val_dataset)} samples')
        if train_val_dataset:
            print(f'Loaded train validation subset: {len(train_val_dataset)} samples')

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    wandb_kwargs = {}
    if hasattr(cfg, 'use_wandb'):
        wandb_kwargs['use_wandb'] = cfg.use_wandb
    if hasattr(cfg, 'wandb_project'):
        wandb_kwargs['wandb_project'] = cfg.wandb_project
    if hasattr(cfg, 'wandb_name'):
        wandb_kwargs['wandb_name'] = cfg.wandb_name
    if hasattr(cfg, 'wandb_config'):
        wandb_kwargs['wandb_config'] = cfg.wandb_config

    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict, dataset, **cfg.trainer.args,
        output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt,
        val_dataset=val_dataset,
        train_val_dataset=train_val_dataset,
        **wandb_kwargs
    )

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory (expands {data_dir} in config paths)')
    parser.add_argument('--auto_retry', type=int, default=0, help='Number of retries on error')
    ## debug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    ## wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='recgen-training', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name (defaults to output_dir name)')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    config = json.load(open(opt.config, 'r'))
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    ## Expand {data_dir} / {config_dir} placeholders in all config paths
    config_dir = os.path.dirname(os.path.abspath(opt.config))
    cfg = edict(expand_path_placeholders(dict(cfg), cfg.data_dir.rstrip('/'), config_dir))
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')
