#!/usr/bin/env python3
"""
Generate a tiny deterministic WebDataset shard for smoke-testing RecGen training.

Each sample carries BOTH stage-1 (sparse structure latent) and stage-2 (SLAT)
keys, so a single shard serves both the stereo flow and the SLAT flow toy
configs (recgen_training/configs/toy/*.json).

The emitted schema matches exactly what the training datasets expect
(see recgen_training/datasets/webdataset_multiview.py::_decode_sample and
TRAINING.md for the full field table):

    <key>.sha256.txt                              64-hex object id
    <key>.ss_latent.npy                           float32 (8, 16, 16, 16)
    <key>.slat_coords.npy                         int32 (N, 3), values in [0, 64)
    <key>.slat_feats.npy                          float32 (N, 8)
    <key>.cond_image_{v:02d}.jpg                  RGB uint8 (H, W, 3)
    <key>.cond_mask_{v:02d}.png                   uint8 mask (>400 nonzero px)
    <key>.cond_mask_full_{v:02d}.png              uint8 mask
    <key>.cond_depth_{v:02d}.png                  uint16 depth in millimeters
    <key>.cond_depth_st_{v:02d}.png               uint16 stereo depth (mm)
    <key>.num_views.json                          int
    <key>.view_indices.json                       list[int]
    <key>.pose_data.json                          list of per-view pose dicts
    <key>.pose_data_median_quantile_5per.json     same (normal-depth variant)
    <key>.pose_data_st_median_quantile_5per.json  same (stereo-depth variant)
    <key>.view_metadata.json                      list of {"intrinsics": 3x3, "visible_fraction": float}
    <key>.aesthetic_score.json                    float
    <key>.dataset_type.txt                        "objectbased"
    <key>.captions.txt                            str

Also writes toy_pose_stats.json (aggregated global_statistics format) computed
from the emitted poses, for use as the dataset's `pose_stats_file`.

Usage:
    python scripts/make_toy_shards.py --out ./data/toy_wds --num-samples 32 --seed 0
"""

import argparse
import io
import json
import os
import tarfile

import numpy as np
from PIL import Image


IMAGE_SIZE = 256
NUM_VIEWS = 4
DEPTH_MM = 1000.0


def random_quaternion(rng):
    """Uniform random unit quaternion (w, x, y, z)."""
    u1, u2, u3 = rng.uniform(size=3)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([w, x, y, z])


def quaternion_to_matrix(q):
    """Rotation matrix from unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_pose_dict(rng):
    """One per-view pose dict with all representations, passing dataset filters."""
    q = random_quaternion(rng)
    R = quaternion_to_matrix(q)
    trans = rng.uniform(-0.3, 0.3, size=3)
    scale = float(rng.uniform(0.8, 1.5))
    # 6D rotation: first two columns of R, column-major
    rot6d = np.concatenate([R[:, 0], R[:, 1]])
    rot9d = R.flatten()
    d = {
        'quat_w': float(q[0]), 'quat_x': float(q[1]),
        'quat_y': float(q[2]), 'quat_z': float(q[3]),
        'trans_x': float(trans[0]), 'trans_y': float(trans[1]), 'trans_z': float(trans[2]),
        'scale': scale,
    }
    for i, v in enumerate(rot6d):
        d[f'rot6d_{i}'] = float(v)
    for i, v in enumerate(rot9d):
        d[f'rot9d_{i}'] = float(v)
    return d


def render_view(rng):
    """Synthetic view: colored disk on gray background + matching mask/depth."""
    h = w = IMAGE_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.uniform(-20, 20), w / 2 + rng.uniform(-20, 20)
    radius = rng.uniform(40, 70)
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2

    image = np.full((h, w, 3), 128, dtype=np.uint8)
    color = rng.integers(0, 255, size=3, dtype=np.uint8)
    image[disk] = color

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[disk] = 255

    # Depth: flat plate at ~1 m inside the mask, 0 (invalid) outside
    depth = np.zeros((h, w), dtype=np.uint16)
    depth[disk] = np.uint16(DEPTH_MM + rng.uniform(-50, 50))

    depth_st = depth.copy()
    noise = rng.integers(-20, 20, size=int(disk.sum()))
    depth_st[disk] = np.clip(depth_st[disk].astype(np.int64) + noise, 1, 65535).astype(np.uint16)

    return image, mask, depth, depth_st


def encode_png(arr):
    buf = io.BytesIO()
    # PIL infers mode 'L' for uint8 and 'I;16' for uint16 arrays
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def encode_jpg(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def encode_npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def component_stats(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    if values.ndim == 1:
        return {'mean': float(values.mean()), 'std': float(std), 'count': len(values)}
    return {'mean': values.mean(axis=0).tolist(), 'std': std.tolist(), 'count': len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=str, default='./data/toy_wds', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--num-shards', type=int, default=4,
                        help='Number of tar shards. Must be >= the total number of '
                             'training ranks: WebDataset splits shards across ranks, '
                             'and a rank with zero shards deadlocks DDP.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    per_shard = (args.num_samples + args.num_shards - 1) // args.num_shards
    all_poses = []
    tars = [tarfile.open(os.path.join(args.out, f'shard-{i:06d}.tar'), 'w')
            for i in range(args.num_shards)]
    try:
        for idx in range(args.num_samples):
            tar = tars[idx // per_shard]
            key = f'toy{idx:06d}'
            members = {}

            members['sha256.txt'] = (''.join(rng.choice(list('0123456789abcdef'), size=64))).encode()

            # Stage-1 latent
            members['ss_latent.npy'] = encode_npy(rng.standard_normal((8, 16, 16, 16)).astype(np.float32))

            # Stage-2 sparse latent: a random blob of voxels
            n_vox = int(rng.integers(200, 1200))
            coords = rng.integers(8, 56, size=(n_vox, 3)).astype(np.int32)
            coords = np.unique(coords, axis=0)
            members['slat_coords.npy'] = encode_npy(coords)
            members['slat_feats.npy'] = encode_npy(rng.standard_normal((coords.shape[0], 8)).astype(np.float32))

            pose_list = []
            pose_list_st = []
            view_metadata = []
            for v in range(NUM_VIEWS):
                image, mask, depth, depth_st = render_view(rng)
                members[f'cond_image_{v:02d}.jpg'] = encode_jpg(image)
                members[f'cond_mask_{v:02d}.png'] = encode_png(mask)
                members[f'cond_mask_full_{v:02d}.png'] = encode_png(mask)
                members[f'cond_depth_{v:02d}.png'] = encode_png(depth)
                members[f'cond_depth_st_{v:02d}.png'] = encode_png(depth_st)

                pose = make_pose_dict(rng)
                pose_st = dict(pose)  # same pose for the stereo-depth variant
                pose_list.append(pose)
                pose_list_st.append(pose_st)
                all_poses.append(pose)

                view_metadata.append({
                    'intrinsics': [
                        [500.0, 0.0, IMAGE_SIZE / 2],
                        [0.0, 500.0, IMAGE_SIZE / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    'visible_fraction': 0.9,
                })

            members['num_views.json'] = json.dumps(NUM_VIEWS).encode()
            members['view_indices.json'] = json.dumps(list(range(NUM_VIEWS))).encode()
            members['pose_data.json'] = json.dumps(pose_list).encode()
            members['pose_data_median_quantile_5per.json'] = json.dumps(pose_list).encode()
            members['pose_data_st_median_quantile_5per.json'] = json.dumps(pose_list_st).encode()
            members['view_metadata.json'] = json.dumps(view_metadata).encode()
            members['aesthetic_score.json'] = json.dumps(5.5).encode()
            members['dataset_type.txt'] = b'objectbased'
            members['captions.txt'] = b'toy object'

            for name, data in members.items():
                info = tarfile.TarInfo(name=f'{key}.{name}')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    finally:
        for tar in tars:
            tar.close()

    # Aggregated pose stats over the emitted poses (same format as the
    # per-dataset pose_stats_<variant>.json files)
    quats = [[p['quat_x'], p['quat_y'], p['quat_z'], p['quat_w']] for p in all_poses]
    rot6ds = [[p[f'rot6d_{i}'] for i in range(6)] for p in all_poses]
    rot9ds = [[p[f'rot9d_{i}'] for i in range(9)] for p in all_poses]
    transs = [[p['trans_x'], p['trans_y'], p['trans_z']] for p in all_poses]
    scales = [p['scale'] for p in all_poses]

    stats = {
        'version': '1.0',
        'total_samples': len(all_poses),
        'total_filtered': 0,
        'total_skipped': 0,
        'pose_variant': 'median_quantile_5per',
        'global_statistics': {
            'quaternion': component_stats(quats),
            '6d_rotation': component_stats(rot6ds),
            '9d_rotation': component_stats(rot9ds),
            'translation': component_stats(transs),
            'scale': component_stats(scales),
        },
    }
    stats_path = os.path.join(args.out, 'toy_pose_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'Wrote {args.num_samples} samples across {args.num_shards} shards to {args.out}')
    print(f'Wrote pose stats to {stats_path}')


if __name__ == '__main__':
    main()
