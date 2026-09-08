"""
Toy-shard schema test (CPU): generate the toy WebDataset shard and verify both
release dataset classes decode it into correctly shaped training samples, and
that the collate functions produce valid batches.
"""

import os
import subprocess
import sys

import pytest
import torch

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, RELEASE_ROOT)


@pytest.fixture(scope='module')
def toy_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp('toy_wds')
    subprocess.run(
        [sys.executable, os.path.join(RELEASE_ROOT, 'scripts', 'make_toy_shards.py'),
         '--out', str(out), '--num-samples', '8', '--seed', '0'],
        check=True,
    )
    return out


def _common_args(toy_dir):
    return dict(
        shards=os.path.join(str(toy_dir), 'shard-{000000..000003}.tar'),
        num_views=2, image_size=518, shuffle=False, shardshuffle=False,
        interleave_shards=False, length=8,
        require_pose_data=True, require_pointmap=True,
        use_pose_normalization=True, pose_variant='median_quantile_5per',
        mix_stereo_depth=True,
        pose_stats_file=os.path.join(str(toy_dir), 'toy_pose_stats.json'),
    )


def test_stereo_toy_samples(toy_dir):
    from recgen_training.datasets import WebDatasetMultiViewSparseStructureLatent

    ds = WebDatasetMultiViewSparseStructureLatent(**_common_args(toy_dir))
    samples = []
    for s in ds:
        samples.append(s)
        if len(samples) == 4:
            break
    assert len(samples) == 4, 'toy samples were filtered out unexpectedly'
    for s in samples:
        assert s['x_0'].shape == (8, 16, 16, 16)
        assert s['cond'].shape == (2, 3, 518, 518)
        assert s['cond_mask'].shape == (2, 1, 518, 518)
        assert s['cond_pointmap'].shape == (2, 3, 518, 518)
        assert s['pose_10d'].shape == (2, 10)
        assert s['pose_13d'].shape == (2, 13)
        assert torch.isfinite(s['cond_pointmap']).all()
        assert s['depth_type'] in ('st', 'normal')

    batch = WebDatasetMultiViewSparseStructureLatent.collate_fn(samples)
    assert batch['x_0'].shape == (4, 8, 16, 16, 16)
    assert batch['cond'].shape == (4, 2, 3, 518, 518)


def test_slat_toy_samples(toy_dir):
    from recgen_training.datasets import WebDatasetMultiViewStructuredLatent

    args = _common_args(toy_dir)
    args['normalization'] = {'mean': [0.0] * 8, 'std': [1.0] * 8}
    args['min_aesthetic_score'] = 4.5
    ds = WebDatasetMultiViewStructuredLatent(**args)
    samples = []
    for s in ds:
        samples.append(s)
        if len(samples) == 4:
            break
    assert len(samples) == 4
    for s in samples:
        assert s['coords'].ndim == 2 and s['coords'].shape[1] == 3
        assert s['feats'].shape == (s['coords'].shape[0], 8)
        assert s['cond'].shape == (2, 3, 518, 518)

    packs = WebDatasetMultiViewStructuredLatent.collate_fn(samples, split_size=2)
    assert len(packs) == 2
    for pack in packs:
        assert hasattr(pack['x_0'], 'coords') and hasattr(pack['x_0'], 'feats')
        assert pack['cond'].shape[0] == 2


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
