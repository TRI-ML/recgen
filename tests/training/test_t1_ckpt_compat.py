"""
T1 — Checkpoint compatibility (CPU).

The released EMA checkpoints must load with strict=True into the release model
classes instantiated from the released config JSONs, with all-finite weights.

Requires the released checkpoints locally (RECGEN_RELEASED_CKPTS, default
./checkpoints/RecGen — see scripts/download_pretrained.py); skipped when unavailable.
"""

import json
import os
import sys

import pytest
import torch

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, RELEASE_ROOT)

CKPT_DIR = os.environ.get('RECGEN_RELEASED_CKPTS', os.path.join(RELEASE_ROOT, 'checkpoints', 'RecGen'))

CASES = [
    ('stereo_config.json', 'stereo_denoiser_ema0.9999_step0055000.pt'),
    ('sparse-structure-ft-70k/stereo_config.json',
     'sparse-structure-ft-70k/stereo_denoiser_ema0.9999_step0070000.pt'),
    ('slat_config.json', 'slat_denoiser_ema0.9999_step0075000.pt'),
]


@pytest.mark.parametrize('config_name,ckpt_name', CASES)
def test_released_ckpt_strict_load(config_name, ckpt_name):
    if not os.path.isdir(CKPT_DIR):
        pytest.skip(f'released checkpoints not found at {CKPT_DIR}')
    if not os.path.exists(os.path.join(CKPT_DIR, ckpt_name)):
        pytest.skip(f'{ckpt_name} not found under {CKPT_DIR}')

    from recgen_inference.recgen_modules import models

    with open(os.path.join(CKPT_DIR, config_name)) as f:
        cfg = json.load(f)
    model_cfg = cfg['models']['denoiser']
    model = getattr(models, model_cfg['name'])(**model_cfg['args'])

    state = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True), None
    # load_state_dict(strict=True) raises on mismatch; reaching here means full match

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f'non-finite values in {name}'

    n_params = sum(p.numel() for p in model.parameters())
    print(f'{model_cfg["name"]}: strict load OK, {n_params/1e6:.1f}M params, all finite')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
