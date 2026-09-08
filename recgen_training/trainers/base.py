from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable Weights & Biases logging.")

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler

def to_jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.ndim == 0 else x.tolist()
    # optionally handle sets, Path, etc.
    if isinstance(x, set):
        return list(x)
    return str(x)  # safe fallback

class Trainer:
    """
    Base class for training.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        val_dataset=None,
        train_val_dataset=None,
        use_wandb=False,
        wandb_project=None,
        wandb_name=None,
        wandb_config=None,
        snapshot_batch_size=4,
        enable_snapshot=True,
        num_workers=8,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.train_val_dataset = train_val_dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.snapshot_batch_size = snapshot_batch_size
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.enable_snapshot = enable_snapshot
        self.num_workers = num_workers
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_config = wandb_config

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))
            
            # Initialize Weights & Biases
            if self.use_wandb:
                # Prepare wandb config
                wandb_cfg = {
                    'batch_size': self.batch_size,
                    'batch_size_per_gpu': self.batch_size_per_gpu,
                    'batch_split': self.batch_split,
                    'max_steps': self.max_steps,
                    'optimizer': self.optimizer_config,
                    'lr_scheduler': self.lr_scheduler_config,
                    'grad_clip': self.grad_clip,
                    'ema_rate': self.ema_rate,
                    'fp16_mode': self.fp16_mode,
                }
                # Merge with user-provided config
                if self.wandb_config is not None:
                    wandb_cfg.update(self.wandb_config)
                
                # Initialize wandb
                wandb.init(
                    project=self.wandb_project or 'recgen-training',
                    name=self.wandb_name or os.path.basename(self.output_dir),
                    config=wandb_cfg,
                    dir=self.output_dir,
                    resume='allow',
                    id=self.wandb_name or os.path.basename(self.output_dir),
                )
                print(f'\nWeights & Biases initialized: {wandb.run.url}')

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        is_iterable = isinstance(self.dataset, IterableDataset)

        # For IterableDataset (e.g., WebDataset), sampler must be None.
        if is_iterable:
            self.data_sampler = None
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
                sampler=None,
            )
        else:
            self.data_sampler = ResumableSampler(
                self.dataset,
                shuffle=True,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=min(int(np.ceil(os.cpu_count() / torch.cuda.device_count())), 12),
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
                sampler=self.data_sampler,
            )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(self.dataset, IterableDataset)
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=False if is_iterable else True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=None, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if batch_size is None:
            batch_size = self.snapshot_batch_size

        if self.is_master:
            print(f'\nSampling {num_samples} images (batch_size={batch_size})...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                elif isinstance(vis, torch.Tensor) and vis.dim() == 4:
                    samples[key] = {'value': vis, 'type': 'image'}
                else:
                    # visualize_sample returned raw data that can't be saved as image
                    # (e.g. 5D voxel grids from SS models without a visualizer)
                    del samples[key]

        # Gather results
        if self.world_size > 1:
            # Synchronize keys: only gather keys present on ALL ranks.
            # Some keys (e.g. mesh overlay images) are master-only; non-master ranks
            # skip rendering to avoid Open3D EGL hangs in multi-process environments.
            # Gathering only common keys prevents dist.gather/all_reduce deadlocks.
            local_keys = list(samples.keys())
            all_keys_obj = [None] * self.world_size
            dist.gather_object(local_keys, all_keys_obj if self.is_master else None, dst=0)
            if self.is_master:
                common_keys = sorted(set.intersection(*[set(k) for k in all_keys_obj]))
            else:
                common_keys = None
            common_keys_obj = [common_keys]
            dist.broadcast_object_list(common_keys_obj, src=0)
            gather_keys = common_keys_obj[0]

            for key in gather_keys:
                # For scalar metrics, use all_reduce to average across GPUs
                if samples[key]['type'] == 'metric':
                    metric_tensor = torch.tensor([samples[key]['value']], device=self.device)
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    samples[key]['value'] = (metric_tensor.item() / self.world_size)
                    continue
                # NCCL backend requires tensors on GPU for gather
                val_gpu = samples[key]['value'].contiguous().to(self.device)
                if self.is_master:
                    all_images = [torch.empty_like(val_gpu) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(val_gpu, all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples].cpu()

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            
            # Prepare data for wandb logging
            if self.use_wandb:
                # Organize samples by index for easy comparison
                max_log = min(16, num_samples)  # Limit to avoid spam
                sample_by_idx = {}  # {idx: {key: image_array}}
                
                # Collect all images by sample index
                for key in samples.keys():
                    if samples[key]['type'] == 'image':
                        # Save grid to disk (keep existing behavior)
                        utils.save_image(
                            samples[key]['value'],
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                            nrow=int(np.sqrt(num_samples)),
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )
                        
                        # Normalize images for display
                        images = samples[key]['value'].clone()
                        if self.dataset.value_range is not None:
                            vmin, vmax = self.dataset.value_range
                            images = (images - vmin) / (vmax - vmin)
                        images = torch.clamp(images, 0, 1)
                        
                        # Store each image by index
                        for i in range(max_log):
                            if i not in sample_by_idx:
                                sample_by_idx[i] = {}
                            
                            img = images[i].cpu().numpy()
                            # Convert CHW to HWC
                            if img.ndim == 3:
                                img = img.transpose(1, 2, 0)
                            # Handle grayscale
                            if img.shape[-1] == 1:
                                img = img.squeeze(-1)
                            
                            sample_by_idx[i][key] = img
                    
                    elif samples[key]['type'] == 'number':
                        min_val = samples[key]['value'].min()
                        max_val = samples[key]['value'].max()
                        images = (samples[key]['value'] - min_val) / (max_val - min_val)
                        images = utils.make_grid(
                            images,
                            nrow=int(np.sqrt(num_samples)),
                            normalize=False,
                        )
                        save_image_with_notes(
                            images,
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                            notes=f'{key} min: {min_val}, max: {max_val}',
                        )
                
                # Create wandb Table for side-by-side comparison
                if sample_by_idx:
                    # Get all unique keys (sample types)
                    all_keys = set()
                    for idx_samples in sample_by_idx.values():
                        all_keys.update(idx_samples.keys())
                    all_keys = sorted(list(all_keys))
                    
                    # Create table with columns: [id] + [key1, key2, ...]
                    # Important: wandb.Table expects wandb.Image objects to be created inline with the data
                    columns = ["sample_id"] + all_keys
                    table = wandb.Table(columns=columns)
                    
                    for idx in sorted(sample_by_idx.keys()):
                        row = [idx]
                        for key in all_keys:
                            if key in sample_by_idx[idx]:
                                # Add wandb.Image directly to the row
                                row.append(wandb.Image(sample_by_idx[idx][key]))
                            else:
                                row.append(None)
                        table.add_data(*row)
                    
                    # Log table
                    wandb.log({f"samples_comparison/{suffix}": table}, step=self.step)
                    
                    # Also log individual images for each key (for easier filtering in wandb UI)
                    for key in all_keys:
                        images_list = []
                        for idx in sorted(sample_by_idx.keys()):
                            if key in sample_by_idx[idx]:
                                images_list.append(wandb.Image(sample_by_idx[idx][key], caption=f"sample_{idx}"))
                        if images_list:
                            wandb.log({f"samples/{key}": images_list}, step=self.step)
                
                # Log scalar metrics
                metrics_to_log = {}
                for key in samples.keys():
                    if samples[key]['type'] == 'metric':
                        metrics_to_log[f"snapshot/{key}"] = samples[key]['value']
                if metrics_to_log:
                    print(f"Logging metrics to wandb: {metrics_to_log}")
                    wandb.log(metrics_to_log, step=self.step)
                else:
                    print("No metrics found in samples dict for wandb logging")
            else:
                # No wandb, just save to disk
                for key in samples.keys():
                    if samples[key]['type'] == 'image':
                        utils.save_image(
                            samples[key]['value'],
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                            nrow=int(np.sqrt(num_samples)),
                            normalize=True,
                            value_range=self.dataset.value_range,
                        )
                    elif samples[key]['type'] == 'number':
                        min_val = samples[key]['value'].min()
                        max_val = samples[key]['value'].max()
                        images = (samples[key]['value'] - min_val) / (max_val - min_val)
                        images = utils.make_grid(
                            images,
                            nrow=int(np.sqrt(num_samples)),
                            normalize=False,
                        )
                        save_image_with_notes(
                            images,
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                            notes=f'{key} min: {min_val}, max: {max_val}',
                        )

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                # Get batch size from the first TENSOR field in the data dict.
                # We must skip non-tensor fields like strings, because:
                #   - data dict may contain: {'sha256': ['abc', 'def'], 'x_0': tensor([4,8,16,16,16]), ...}
                #   - Strings don't have .shape, so v[0].shape would crash
                #   - We only need batch_size from actual tensor data
                batch_size = None
                for v in data.values():
                    if isinstance(v, (tuple, list)):
                        # Check if first element is a tensor (has .shape)
                        if len(v) > 0 and hasattr(v[0], 'shape'):
                            batch_size = v[0].shape[0]
                            break
                        # else: skip this field (it's a list of strings/bytes/etc)
                    elif hasattr(v, 'shape'):
                        # It's a tensor
                        batch_size = v.shape[0]
                        break
                    # else: skip this field (it's a scalar or string)
                
                if batch_size is None:
                    raise ValueError("Could not determine batch size - no tensor fields found in data")

                def split_value(v, start, end):
                    """Helper to split a value for batch_split.
                    - Tensors: slice them v[start:end]
                    - Lists/tuples of tensors: slice each element
                    - Lists/tuples of non-tensors (strings, etc): slice the list/tuple itself
                    - Strings/scalars: return as-is (can't be split)
                    """
                    if isinstance(v, tuple):
                        # Check if it's a tuple of non-tensors
                        if len(v) > 0 and not hasattr(v[0], 'shape'):
                            return v[start:end]
                        # Otherwise it's a tuple of tensors, slice each element
                        return tuple(
                            elem[start:end] if hasattr(elem, 'shape') else elem
                            for elem in v
                        )
                    elif isinstance(v, list):
                        # Check if it's a list of non-tensors (like dataset_type strings)
                        if len(v) > 0 and not hasattr(v[0], 'shape'):
                            return v[start:end]
                        # Otherwise it's a list of tensors, slice each element
                        return [
                            elem[start:end] if hasattr(elem, 'shape') else elem
                            for elem in v
                        ]
                    elif hasattr(v, 'shape'):
                        # It's a tensor - slice it
                        return v[start:end]
                    else:
                        # Scalar/string - return as-is
                        return v
                
                data_list = [
                    {k: split_value(v, i * batch_size // self.batch_split, (i + 1) * batch_size // self.batch_split) 
                     for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            try:
                self.snapshot_dataset()
            except Exception as e:
                print(f"Warning: Failed to create dataset snapshot: {e}")
        if self.enable_snapshot:
            if self.step == 0:
                try:
                    self.snapshot(suffix='init')
                except Exception as e:
                    print(f"Warning: Failed to create init snapshot: {e}")
            else: # resume
                try:
                    self.snapshot(suffix=f'resume_step{self.step:07d}')
                except Exception as e:
                    print(f"Warning: Failed to create resume snapshot: {e}")

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        time_data_total = 0.0
        time_step_total = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            time_after_data = time.time()

            step_log = self.run_step(data_list)
            torch.cuda.synchronize()

            time_end = time.time()
            time_elapsed += time_end - time_start
            time_data_total += time_after_data - time_start
            time_step_total += time_end - time_after_data

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                data_pct = time_data_total / max(time_elapsed, 1e-9) * 100
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                    f'Data: {data_pct:.1f}%',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.enable_snapshot and self.step % self.i_sample == 0:
                try:
                    self.snapshot()
                except Exception as e:
                    print(f"Warning: Failed to create snapshot at step {self.step}: {e}")
                torch.cuda.empty_cache()

            # Save checkpoint
            if self.step % self.i_save == 0:
                if self.is_master:
                    self.save()
                if self.world_size > 1:
                    dist.barrier()

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(log, default=to_jsonable)}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with tensorboard and wandb
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    if len(log_show) == 0:
                        print(f'WARNING: All {len(log)} log entries contain NaN at step {self.step}, skipping logging')
                    else:
                        log_show = dict_reduce(log_show, lambda x: np.mean(x))
                        log_show = dict_flatten(log_show, sep='/')
                        for key, value in log_show.items():
                            self.writer.add_scalar(key, value, self.step)

                        # Log to wandb
                        if self.use_wandb:
                            wandb.log(log_show, step=self.step)


                    log = []

        if self.is_master:
            self.writer.close()
            if self.use_wandb:
                wandb.finish()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            