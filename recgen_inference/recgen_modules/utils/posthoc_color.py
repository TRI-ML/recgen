"""Post-hoc color correction of generated assets against the input image.

All fitters take paired pixels sampled at the SAME locations:
  src: rendered predicted-asset colors, float (N, 3) in [0, 1]
  tgt: input-image colors at those pixels, float (N, 3) in [0, 1]
and return a transform dict (or None if the fit is unreliable / not an
improvement). `apply_transform` maps asset colors (M, 3) in [0, 1] -> corrected.

Design mirrors the v3 dataset correction: luminance-preserving-hue gamma as the
primary family, quality gates so unreliable fits fall back to identity, clamps
as the "avoid extreme exposure artifacts" knob.
"""
import numpy as np

DARK_T = 0.05          # ignore pixels where the rendered asset is near-black
MIN_PX = 200
MIN_IOU = 0.25   # render/mask alignment gate: below this the overlap pixels are
                 # not in correspondence and an L1-improving fit can still be a
                 # degenerate wash toward the photo mean
TRIM_Q = 0.8
GAMMA_LO, GAMMA_HI = 0.5, 2.0
SCALE_LO, SCALE_HI = 0.3, 3.0
OFF_LIM = 0.5
MIN_REL_GAIN = 0.03    # require >=3% L1 improvement on the fit pixels


def _lum(x):
    return 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]


def _prep(src, tgt):
    src = np.clip(np.asarray(src, np.float64), 0, 1)
    tgt = np.clip(np.asarray(tgt, np.float64), 0, 1)
    keep = _lum(src) > DARK_T
    if keep.sum() < MIN_PX:
        return None
    return src[keep], tgt[keep]


def _trim(src, tgt, pred):
    resid = np.abs(pred - tgt).mean(1)
    keep = resid <= np.quantile(resid, TRIM_Q)
    return (src[keep], tgt[keep]) if keep.sum() >= MIN_PX else (src, tgt)


def _gate(src, tgt, pred):
    """Accept only if the transform meaningfully reduces L1 on fit pixels."""
    before = np.abs(src - tgt).mean()
    after = np.abs(pred - tgt).mean()
    return after <= before * (1.0 - MIN_REL_GAIN)


def fit_gamma_lum(src, tgt):
    """Single luminance gamma applied to all channels (hue preserving)."""
    from scipy.optimize import minimize_scalar
    p = _prep(src, tgt)
    if p is None:
        return None
    s, t = p
    ls, lt = np.clip(_lum(s), 1e-6, 1), _lum(t)

    def fit(a, b):
        return float(minimize_scalar(lambda g: np.mean((np.power(a, g) - b) ** 2),
                                     bounds=(0.2, 4.0), method='bounded').x)
    g0 = fit(ls, lt)
    s2, t2 = _trim(s, t, np.power(np.clip(s, 1e-6, 1), g0))
    g = fit(np.clip(_lum(s2), 1e-6, 1), _lum(t2))
    if not (GAMMA_LO <= g <= GAMMA_HI):
        return None
    tr = {'type': 'gamma', 'gamma': g}
    return tr if _gate(s, t, apply_transform(s, tr)) else None


def fit_affine_pc(src, tgt):
    """Per-channel affine: c' = scale*c + off, clamped."""
    p = _prep(src, tgt)
    if p is None:
        return None
    s, t = p
    def solve(a, b):
        sc, of = np.ones(3), np.zeros(3)
        for i in range(3):
            A = np.vstack([a[:, i], np.ones(len(a))]).T
            (sc[i], of[i]), *_ = np.linalg.lstsq(A, b[:, i], rcond=None)
        return np.clip(sc, SCALE_LO, SCALE_HI), np.clip(of, -OFF_LIM, OFF_LIM)
    sc, of = solve(s, t)
    s2, t2 = _trim(s, t, np.clip(s * sc + of, 0, 1))
    sc, of = solve(s2, t2)
    tr = {'type': 'affine', 'scale': sc.tolist(), 'off': of.tolist()}
    return tr if _gate(s, t, apply_transform(s, tr)) else None


def fit_gamma_affine(src, tgt):
    """Luminance gamma first, then residual per-channel affine on top."""
    g = fit_gamma_lum(src, tgt)
    base = apply_transform(np.clip(np.asarray(src, np.float64), 0, 1),
                           g) if g else np.clip(np.asarray(src, np.float64), 0, 1)
    a = fit_affine_pc(base, tgt)
    if g is None and a is None:
        return None
    return {'type': 'gamma_affine', 'gamma': (g or {}).get('gamma', 1.0),
            'scale': (a or {'scale': [1, 1, 1]})['scale'],
            'off': (a or {'off': [0, 0, 0]})['off']}


def apply_transform(colors, tr):
    """colors (M,3) in [0,1] -> corrected (M,3) in [0,1]."""
    c = np.clip(np.asarray(colors, np.float64), 0, 1)
    if tr is None:
        return c
    if tr['type'] in ('gamma', 'gamma_affine'):
        c = np.power(np.clip(c, 1e-6, 1), tr.get('gamma', 1.0))
    if tr['type'] in ('affine', 'gamma_affine'):
        c = c * np.asarray(tr['scale']) + np.asarray(tr['off'])
    return np.clip(c, 0, 1)


FITTERS = {'gamma': fit_gamma_lum, 'affine': fit_affine_pc,
           'gamma_affine': fit_gamma_affine}


# ---------------------------------------------------------------------------
# Asset-level calibration against the input view (deployment entry point)
# ---------------------------------------------------------------------------

def _lookat_camera(M, K, S=384, fill=0.7):
    """Similarity canonical->camera + pinhole K -> (rigid extrinsics, normalized
    symmetric intrinsics, 3x3 photo homography). The virtual camera is rotated
    to look at the object so the principal point stays centered (required by the
    gaussian rasterizer's GL projection)."""
    import torch
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    R = M[:3, :3] / s
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    t = M[:3, 3] / s
    z = t / np.linalg.norm(t)
    v = np.cross(z, [0, 0, 1.0])
    w = float(np.dot(z, [0, 0, 1.0]))
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rl = np.eye(3) + Vx + Vx @ Vx / (1 + w)
    E = np.eye(4)
    E[:3, :3] = Rl @ R
    E[:3, 3] = Rl @ t
    zc = float(np.linalg.norm(t))
    fx_n = (fill / 2) / (0.55 / zc)
    In = torch.tensor([[fx_n, 0, 0.5], [0, fx_n, 0.5], [0, 0, 1]],
                      dtype=torch.float32, device='cuda')
    Kv = np.array([[fx_n * S, 0, S / 2], [0, fx_n * S, S / 2], [0, 0, 1]])
    H = Kv @ Rl @ np.linalg.inv(K)
    return torch.tensor(E, dtype=torch.float32, device='cuda'), In, H


def calibrate_asset(gaussian, M_similarity, K, rgb_full, mask_full,
                    variant='gamma_affine', S=384):
    """Fit a color transform mapping the predicted asset's rendered colors to
    the input photo (HB-validated recipe: fit on the input view's mask∩render
    overlap; returns (transform_or_None, stats)). Does NOT modify the asset.

    Args:
        gaussian: Gaussian representation (canonical frame)
        M_similarity: (4,4) canonical->camera similarity (inv(cam2ncam) @ pose)
        K: (3,3) pinhole intrinsics of rgb_full
        rgb_full: (H,W,3) uint8 input image
        mask_full: (H,W) uint8/bool object mask
        variant: 'gamma' | 'affine' | 'gamma_affine'
    """
    import cv2
    import torch
    from ..renderers.gaussian_render import GaussianRenderer

    mask_full = np.asarray(mask_full)
    if mask_full.dtype == bool:
        mask_full = mask_full.astype(np.uint8) * 255
    elif mask_full.max() == 1:
        mask_full = mask_full * 255

    E, In, H = _lookat_camera(np.asarray(M_similarity, np.float64),
                              np.asarray(K, np.float64), S=S)
    photo = cv2.warpPerspective(np.asarray(rgb_full), H, (S, S)).astype(np.float64) / 255
    mask = cv2.warpPerspective(mask_full, H, (S, S), flags=cv2.INTER_NEAREST) > 127

    renderer = GaussianRenderer({'resolution': S, 'near': 0.01, 'far': 20.0,
                                 'ssaa': 1, 'bg_color': (0, 0, 0)})
    with torch.no_grad():
        rend = renderer.render(gaussian, E, In)['color'].permute(1, 2, 0).cpu().numpy()
        n = gaussian._features_dc.reshape(-1, 3).shape[0]
        alpha = renderer.render(gaussian, E, In,
                                colors_overwrite=torch.ones(n, 3, device='cuda')
                                )['color'].mean(0).cpu().numpy()
    cover = alpha > 0.5
    ov = mask & cover
    union = (mask | cover).sum()
    stats = {'overlap_px': int(ov.sum()),
             'iou': float(ov.sum() / union) if union else 0.0}
    if ov.sum() < MIN_PX:
        return None, {**stats, 'reason': 'insufficient overlap'}
    if stats['iou'] < MIN_IOU:
        return None, {**stats, 'reason': 'poor render/mask alignment'}

    src, tgt = rend[ov], photo[ov]
    tr = FITTERS[variant](src, tgt)
    if tr is None:
        return None, {**stats, 'reason': 'fit gated (no reliable improvement)'}
    stats['l1_before'] = float(np.abs(np.clip(src, 0, 1) - tgt).mean())
    stats['l1_after'] = float(np.abs(apply_transform(src, tr) - tgt).mean())
    return tr, stats


def apply_to_gaussian(gaussian, tr):
    """Apply a fitted transform to the gaussian's DC colors in place."""
    import torch
    SH_C0 = 0.28209479177387814
    dc = gaussian._features_dc
    cols = torch.clamp(0.5 + SH_C0 * dc.reshape(-1, 3), 0, 1)
    new = apply_transform(cols.detach().cpu().numpy(), tr)
    gaussian._features_dc = ((torch.tensor(new, dtype=dc.dtype, device=dc.device)
                              - 0.5) / SH_C0).reshape(dc.shape)
