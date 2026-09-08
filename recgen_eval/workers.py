"""Multi-GPU worker orchestration for the evaluation driver.

Workers are plain subprocesses re-invoking
``python -m recgen_eval.run`` with ``--split_size/--split_index``; multiple
workers may share one GPU (``--workers_per_gpu``).
"""

import glob as glob_module
import os
import pickle
import subprocess
import sys


def get_gpu_ids():
    """Get the list of GPU IDs from CUDA_VISIBLE_DEVICES."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        return [gpu.strip() for gpu in cuda_visible.split(',') if gpu.strip()]
    try:
        import torch
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return ['0']


def run_parallel_workers(args, gpu_ids):
    """Spawn workers to run sharded evaluation across GPUs.

    Multiple workers can share the same GPU (controlled by --workers_per_gpu).
    """
    num_gpus = len(gpu_ids)
    workers_per_gpu = args.workers_per_gpu
    total_workers = num_gpus * workers_per_gpu
    print(f"\n{'='*80}")
    print(f"Parallel evaluation: spawning {total_workers} workers ({workers_per_gpu}/GPU) on GPUs {gpu_ids}")
    print(f"{'='*80}\n")

    base_cmd = [sys.executable, '-m', 'recgen_eval.run']
    for key, value in vars(args).items():
        if key in ('parallel', 'split_size', 'split_index', 'workers_per_gpu'):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                base_cmd.append(f'--{key}')
            continue
        if isinstance(value, list):
            base_cmd.append(f'--{key}')
            base_cmd.extend(str(v) for v in value)
            continue
        base_cmd.extend([f'--{key}', str(value)])

    use_pointmap = bool(args.use_pointmap)
    use_predicted_scale = bool(args.use_predicted_scale)
    log_dir = os.path.join(
        args.save_path_root + ("_debug" if args.debug else ""),
        args.experiment_name,
        f"pointmap_{use_pointmap}_predicted_scale_{use_predicted_scale}",
        f"dataset_{args.dataset_name}_multiview{args.num_views}"
    )
    os.makedirs(log_dir, exist_ok=True)

    processes = []
    log_files = []
    worker_idx = 0
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            worker_cmd = base_cmd + [
                '--split_size', str(total_workers),
                '--split_index', str(worker_idx),
                '--parallel', '0',
            ]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id

            log_path = os.path.join(log_dir, f"worker_{worker_idx}.log")
            log_file = open(log_path, 'w')
            log_files.append((worker_idx, log_path, log_file))

            print(f"[Worker {worker_idx}] CUDA_VISIBLE_DEVICES={gpu_id}, split={worker_idx}/{total_workers}, log={log_path}")
            proc = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((worker_idx, gpu_id, proc))
            worker_idx += 1

    failed = []
    for i, gpu_id, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            failed.append((i, gpu_id, proc.returncode))
            print(f"[Worker {i}] FAILED with return code {proc.returncode}")
        else:
            print(f"[Worker {i}] completed successfully")

    for i, log_path, log_file in log_files:
        log_file.close()
        print(f"[Worker {i}] Log saved to: {log_path}")

    if failed:
        raise RuntimeError(f"Parallel evaluation failed. Failed workers: {failed}")

    return total_workers


def merge_split_results(save_path, split_size, use_pointmap, use_predicted_scale, config_file):
    """Merge per-split pickle outputs into a single final_results folder."""
    from .poseval6d.evaluation.evaluator import Evaluator

    predictions_dir = os.path.join(save_path, "predictions")
    output_metrics = os.path.join(predictions_dir, "final_results")

    pattern = os.path.join(predictions_dir, f"predictions_{split_size}_*.pkl")
    split_files = sorted(glob_module.glob(pattern))

    if not split_files:
        print(f"Warning: No split files found matching {pattern}")
        return

    print(f"\n{'='*80}")
    print(f"Merging {len(split_files)} split results into {output_metrics}")
    print(f"{'='*80}\n")

    all_results = []
    for pkl_file in split_files:
        print(f"Loading {pkl_file}")
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
            all_results.extend(results)

    print(f"Total samples after merge: {len(all_results)}")

    merged_pickle = os.path.join(predictions_dir, "predictions_1_0.pkl")
    with open(merged_pickle, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved merged predictions to {merged_pickle}")

    final_evaluator = Evaluator(config=config_file)
    for object_id, metrics, file_name in all_results:
        failed = metrics.get('_failed', False)
        final_evaluator.add_metrics(object_id, metrics, file_name, failed=failed)

    final_evaluator.save_metrics(output_path=output_metrics)
    print(f"Saved merged RECGEN multi-view results to {output_metrics}")
