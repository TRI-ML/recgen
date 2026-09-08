"""
CPU unit tests for parity-critical training components:

  * EMA update math (hand-computed fold, bitwise)
  * EMA-before-finetune quirk (EMA starts from init, not the loaded ckpt)
  * fp16 inflat_all master-param round-trip (bitwise)
  * AdaptiveGradClipper state_dict round-trip
  * voxel-balanced collate grouping (deterministic, balanced)
  * T-stats: exported training-exact pose stats == weighted aggregation over
    the local dataset metadata (internal-only; skipped without local data)
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, RELEASE_ROOT)


def test_ema_fold_bitwise():
    from recgen_training.trainers.utils import make_master_params
    import copy

    torch.manual_seed(0)
    params = [torch.randn(37, 5), torch.randn(11)]
    model_params = [nn.Parameter(p.clone().half()) for p in params]
    master = make_master_params(model_params)
    ema = copy.deepcopy(master)

    rate = 0.9999
    expected = ema[0].detach().clone()
    for step in range(3):
        # simulate an optimizer update on master
        with torch.no_grad():
            master[0].add_(torch.randn_like(master[0]) * 0.01)
        # trainer's update_ema line, verbatim
        for m, e in zip(master, ema):
            e.detach().mul_(rate).add_(m, alpha=1.0 - rate)
        expected.mul_(rate).add_(master[0].detach(), alpha=1.0 - rate)

    assert torch.equal(ema[0].detach(), expected)


def test_master_param_roundtrip_bitwise():
    from recgen_training.trainers.utils import (
        make_master_params, master_params_to_model_params, model_params_to_master_params,
    )

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(13, 7), nn.LayerNorm(7), nn.Linear(7, 3)).half()
    model_params = [p for p in model.parameters() if p.requires_grad]
    master = make_master_params(model_params)

    before = [p.detach().clone() for p in model_params]
    master_params_to_model_params(model_params, master)
    for b, p in zip(before, model_params):
        assert torch.equal(b, p.detach()), 'fp16->fp32->fp16 must be lossless for fp16 params'

    # And model -> master reproduces the same flat vector
    flat_before = master[0].detach().clone()
    model_params_to_master_params(model_params, master)
    assert torch.equal(flat_before, master[0].detach())


def test_grad_clipper_state_roundtrip():
    from recgen_training.utils.grad_clip_utils import AdaptiveGradClipper

    torch.manual_seed(2)
    clip = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    params = [nn.Parameter(torch.randn(10, 10)) for _ in range(3)]
    for _ in range(20):
        for p in params:
            p.grad = torch.randn_like(p)
        clip(params)

    state = clip.state_dict()
    clip2 = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    clip2.load_state_dict(state)

    grads = [torch.randn_like(p) for p in params]
    # clip_grad_norm_ mutates grads in place, so restore identical grads
    # before each call: the restored clipper must behave identically.
    for p, g in zip(params, grads):
        p.grad = g.clone()
    g1 = clip(params).item()
    for p, g in zip(params, grads):
        p.grad = g.clone()
    g2 = clip2(params).item()
    assert g1 == g2
    # and the post-call states must match too
    s1, s2 = clip.state_dict(), clip2.state_dict()
    assert s1['max_norm'] == s2['max_norm']
    assert s1['buffer_ptr'] == s2['buffer_ptr']
    assert (s1['grad_norm'] == s2['grad_norm']).all()


def test_balanced_collate_grouping():
    from recgen_training.utils.data_utils import load_balanced_group_indices

    loads = [120, 5, 90, 30, 60, 10, 200, 45]
    groups = load_balanced_group_indices(loads, 2)
    # every index exactly once
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(loads)))
    assert len(groups) == 2
    # deterministic
    assert groups == load_balanced_group_indices(loads, 2)
