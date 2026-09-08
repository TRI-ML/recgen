"""Real-data tests for the eval release (self-skipping when data is absent).

Env vars:
  RECGEN_EVAL_DATA_ROOT   eval data root containing HB/ and AV_final/
                          (default ./data/eval — see scripts/download_eval_data.py)

Run: pytest tests/eval/test_eval_data.py -v
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, RELEASE_ROOT)

DATA_ROOT = os.environ.get('RECGEN_EVAL_DATA_ROOT',
                           os.path.join(RELEASE_ROOT, 'data', 'eval'))

needs_hb = pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_ROOT, 'HB', 'hb_val_kinect')),
                              reason='HB eval data not found')
needs_av = pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_ROOT, 'AV_final', 'artvip_test_all')),
                              reason='AV_final eval data not found')


def _load_mv(ds, nv):
    from recgen_eval.poseval6d.dataloader import load_multiview_anchor_dataset
    from recgen_eval.run import default_instance_list
    mv, base = load_multiview_anchor_dataset(
        dataset_name=ds, path_to_datasets=DATA_ROOT, save_path='/tmp/recgen_eval_test',
        num_views=nv, split_size=-1, split_index=-1,
        instance_list_path=default_instance_list(ds, nv))
    return mv, base


@needs_hb
@pytest.mark.parametrize('nv,expected', [(1, 538), (2, 538)])
def test_hb_loader_counts(nv, expected):
    mv, base = _load_mv('HB', nv)
    assert len(mv) == expected
    assert base.width == 960 and base.height == 540  # resize_factor 0.5 of 1920x1080


@needs_av
@pytest.mark.parametrize('nv,expected', [(1, 500), (2, 500)])
def test_av_loader_counts(nv, expected):
    mv, base = _load_mv('AV_final', nv)
    assert len(mv) == expected
    assert base.width == 960 and base.height == 540  # resize_factor 0.5 of 1920x1080


@needs_hb
def test_hb_sample_contents():
    mv, _ = _load_mv('HB', 1)
    views, obj_folder = next(iter(mv))
    v = views[0]
    assert v['rgb'].shape == (540, 960, 3)
    assert v['depth'].shape == (540, 960) and v['depth'].dtype == np.float32
    assert v['mask'].shape == (540, 960)
    assert v['pose'].shape == (4, 4)
    assert 0.1 < float(np.median(v['depth'][v['depth'] > 0])) < 5.0  # metres
    assert os.path.basename(obj_folder).startswith('obj_')
