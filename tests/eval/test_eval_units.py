"""Unit tests for recgen_eval metric and aggregation code (CPU, no data needed).

Run: pytest tests/eval/test_eval_units.py  (or python -m pytest)
"""

import json
import os
import sys

import numpy as np
import pytest
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, RELEASE_ROOT)

from recgen_eval.poseval6d.core.geometry import np_transform_pcd
from recgen_eval.poseval6d.core.metrics import (
    chamfer_distance_between_clouds,
    chamfer_distance_gt_mesh,
    compute_adds_gen,
    match_point_counts,
)
from recgen_eval.aggregate import aggregate_from_json, _save_sample_metrics_json, _load_metrics_from_json

CFG = os.path.join(RELEASE_ROOT, 'recgen_eval', 'configs', 'eval_configs.yaml')


def _evaluator():
    from recgen_eval.poseval6d.evaluation.evaluator import Evaluator
    return Evaluator(config=CFG)


def test_np_transform_pcd_fp16_semantics():
    """The fp16 casts are number-critical: outputs must equal explicit fp16 math."""
    rng = np.random.default_rng(0)
    pcd = rng.normal(size=(100, 3))
    r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    t = rng.normal(size=3)
    out = np_transform_pcd(pcd, r, t)
    expected = np.dot(pcd.astype(np.float16), r.astype(np.float16).T) + t.astype(np.float16)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out, expected)


def test_chamfer_between_clouds_exact():
    pts1 = np.array([[0.0, 0, 0], [1, 0, 0]])
    pts2 = np.array([[0.0, 0, 0.1], [1, 0, 0.3]])
    # NN dists both directions: [0.1, 0.3] each way -> 0.5*(0.2+0.2)=0.2
    assert chamfer_distance_between_clouds(pts1, pts2) == pytest.approx(0.2)


def test_chamfer_identical_mesh_near_zero():
    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    pose = np.eye(4)
    d = chamfer_distance_gt_mesh(pose, mesh, pose, mesh, use_icp=False)
    # identical surfaces, 99999 samples each -> only sampling noise (cm units)
    assert d < 0.5  # < 5 mm on a 20 cm box


def test_chamfer_translation_offset():
    """A known rigid offset must show up in the no-ICP chamfer and be mostly
    removed by ICP."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    gt_pose = np.eye(4)
    pred_pose = np.eye(4)
    pred_pose[:3, 3] = [0.01, 0.0, 0.0]  # 1 cm offset
    d_no_icp = chamfer_distance_gt_mesh(gt_pose, mesh, pred_pose, mesh, use_icp=False)
    d_icp = chamfer_distance_gt_mesh(gt_pose, mesh, pred_pose, mesh, use_icp=True)
    assert 0.2 < d_no_icp < 1.1   # ~<=1 cm effect in cm units (sphere NN < offset)
    assert d_icp < d_no_icp
    assert d_icp < 0.2


def test_match_point_counts():
    a, b = np.zeros((10, 3)), np.zeros((4, 3))
    a2, b2 = match_point_counts(a, b)
    assert len(a2) == len(b2) == 4


def test_compute_adds_gen_identity():
    pts = np.random.default_rng(1).normal(size=(50, 3))
    pose = np.eye(4)
    assert compute_adds_gen(pts, pts, pose, pose) == pytest.approx(0.0, abs=1e-6)


def test_evaluator_anchor_metrics_consistency():
    """chamfer_normalized == chamfer/diam_cm; addss == no_icp/diam_cm; recalls follow."""
    ev = _evaluator()
    gt_mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    pred_mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    gt_pose = np.eye(4)
    pred_pose = np.eye(4)
    meta = {'diameter': float(np.sqrt(3) * 0.2)}
    m = ev.compute_metrics_anchor(3, pred_pose, pred_mesh, gt_pose, gt_mesh, meta)
    diam_cm = meta['diameter'] * 100
    assert m['chamfer_normalized'] == pytest.approx(m['chamfer_dist'] / diam_cm)
    assert m['addss'] == pytest.approx(m['chamfer_dist_no_icp'] / diam_cm)
    assert m['addss_10'] == float(m['addss'] < 0.10)
    assert m['addss_05'] == float(m['addss'] < 0.05)
    assert m['addss_02'] == float(m['addss'] < 0.02)
    assert m['adds_thres'] in (0.0, 1.0)
    assert np.isnan(m['mean_ar'])  # BOP suite not computed in the anchor path
    # config has compute_color_chamfer: true
    assert np.isfinite(m['chamfer_dist_color'])


def test_aggregate_from_json_roundtrip(tmp_path):
    """Fabricated obj_* tree -> aggregate CSVs with correct MEAN (incl. failure penalty)."""
    run_dir = tmp_path / 'dataset_HB_multiview1'
    samples = [
        # (folder, chamfer, addss, failed)
        ('obj_000001_multiview_000001_imgs_000004', 1.0, 0.02, False),
        ('obj_000001_multiview_000001_imgs_000010', 3.0, 0.06, False),
        ('obj_000002_multiview_000002_imgs_000005', None, None, True),
    ]
    for folder, cham, addss, failed in samples:
        d = run_dir / folder
        d.mkdir(parents=True)
        if failed:
            metrics = {'_failed': True, 'adds_thres': 0.0,
                       'chamfer_dist': 20.0, 'chamfer_dist_no_icp': 20.0,
                       'chamfer_dist_icp_with_scale': 20.0}
        else:
            metrics = {'_failed': False, 'chamfer_dist': cham,
                       'chamfer_dist_no_icp': cham, 'chamfer_normalized': cham / 10,
                       'addss': addss,
                       'addss_10': float(addss < 0.1), 'addss_05': float(addss < 0.05),
                       'addss_02': float(addss < 0.02), 'adds_thres': 1.0}
        img = folder.split('_imgs_')[1][:6]
        _save_sample_metrics_json(metrics, str(d), f'{img}.png', failed=failed)

    n = aggregate_from_json(str(run_dir))
    assert n == 3
    import pandas as pd
    stats = pd.read_csv(run_dir / 'predictions' / 'final_results_from_json' /
                        '0_all_frames_metrics_results_statistics.csv')
    mean = stats[stats['Statistic'] == 'MEAN'].iloc[0]
    # chamfer mean over all 3 incl. the 20.0 penalty
    assert mean['CHAMFER'] == pytest.approx((1.0 + 3.0 + 20.0) / 3)
    # addss mean over the 2 successful samples (penalty rows have no addss)
    assert mean['ADDSS'] == pytest.approx((0.02 + 0.06) / 2)
    allf = pd.read_csv(run_dir / 'predictions' / 'final_results_from_json' /
                       '0_all_frames_metrics_results.csv')
    assert len(allf) == 4  # 3 samples + MEAN row
    assert int(allf[allf['Frame_ID'] != 'MEAN']['FAILED'].sum()) == 1
    failed_csv = run_dir / 'predictions' / 'final_results_from_json' / '0_failed_samples.csv'
    assert failed_csv.exists()


def test_metrics_json_roundtrip(tmp_path):
    """NaN -> null -> NaN roundtrip through metrics.json."""
    d = tmp_path / 'obj_000004_multiview_000013_imgs_000040'
    d.mkdir()
    metrics = {'chamfer_dist': 1.5, 'addss': 0.03, 'err_R': np.nan, 'adds_thres': 1.0}
    _save_sample_metrics_json(metrics, str(d), '000040.png',
                              pred_pose=np.eye(4), gt_pose=np.eye(4),
                              gt_metadata={'diameter': 0.25})
    obj_id, loaded, file_name, failed = _load_metrics_from_json(str(d / 'metrics.json'))
    assert obj_id == 4 and file_name == '000040.png' and not failed
    assert loaded['chamfer_dist'] == 1.5
    assert np.isnan(loaded['err_R'])
    raw = json.load(open(d / 'metrics.json'))
    assert raw['err_R'] is None and raw['gt_diameter'] == 0.25


def test_default_instance_lists_packaged():
    from recgen_eval.run import default_instance_list
    hb1 = default_instance_list('HB', 1)
    hb2 = default_instance_list('HB', 2)
    av1 = default_instance_list('AV_final', 1)
    assert hb1.endswith('instance_list_random.txt') and os.path.exists(hb1)
    assert hb2.endswith('instance_list_simple.txt') and os.path.exists(hb2)
    assert os.path.exists(av1)
    assert sum(1 for line in open(hb1) if line.strip()) == 538
    assert sum(1 for line in open(hb2) if line.strip()) == 538
    assert sum(1 for line in open(av1) if line.strip()) == 500
    with pytest.raises(ValueError):
        default_instance_list('LMO', 1)


def test_generate_restrictions():
    """The eval release only ships the paper configuration."""
    from recgen_eval.generate import crop_to_bounding_box
    from PIL import Image
    pm = np.zeros((8, 8, 3), dtype=np.float32)
    img = Image.new('RGB', (8, 8))
    mask = Image.new('L', (8, 8))
    valid = np.zeros((8, 8), dtype=bool)
    with pytest.raises(NotImplementedError):
        crop_to_bounding_box(pm, img, mask, valid, normalization_method='minmax')
    with pytest.raises(NotImplementedError):
        crop_to_bounding_box(pm, img, mask, valid, normalization_method='median_quantile',
                             filter_outliers_type='dbscan')
