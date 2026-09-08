"""RecGenResult dataclass and its .save() method."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import trimesh
from PIL import Image


@dataclass
class RecGenResult:
    """Result of a single RecGen inference.

    Attributes:
        mesh: Final textured mesh in the camera frame.
        raw_mesh: Mesh before the pose transform (object-centric frame).
        pose_matrix: (4, 4) float64 transformation matrix (rotation * scale, then translation).
        pose_quat: (7,) float array with [tx, ty, tz, qx, qy, qz, qw] — scipy convention.
        pose: Dict with fine-grained pose components ('quaternion', 'translation',
            'scale', 'rotation_matrix') as returned by `utils.parse_pose`.
        cam2ncam: (4, 4) float32 normalisation transform applied before running the model.
        pose_representation: Resolved pose representation string (e.g. 'quaternion_translation_scale').
        rgb: (H, W, 3) uint8 input RGB image (anchor view for multi-view).
        intrinsics: (3, 3) camera intrinsics matching `rgb`.
        _outputs: Raw pipeline outputs dict — power-user escape hatch, not part of the
            public API. Prefer `pose_matrix` / `pose_quat` / `mesh` for normal use.
    """

    mesh: trimesh.Trimesh
    raw_mesh: trimesh.Trimesh
    pose_matrix: np.ndarray
    pose_quat: np.ndarray
    pose: Dict[str, Any]
    cam2ncam: np.ndarray
    pose_representation: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    _outputs: Dict[str, Any] = field(repr=False, default_factory=dict)
    posthoc_color: Optional[Dict[str, Any]] = None

    def save(
        self,
        output_dir: Union[str, Path],
        *,
        save_splat: bool = True,
        save_glb: bool = False,
        save_inputs: bool = False,
    ) -> None:
        """Persist the reconstruction to `output_dir`.

        Writes:
            - mesh.obj, posed_mesh.obj (always, with vertex colors baked into MTL)
            - overlay.png (always; posed mesh rendered over the input RGB)
            - turntable.mp4 (always when CUDA renderers are available; gaussian color | mesh normals)
            - metadata.json (always)
            - gaussian.ply, posed_gaussian.ply (if save_splat and the pipeline produced a Gaussian splat)
            - textured_mesh.glb (if save_glb and nvdiffrast/xatlas/pymeshfix are installed)
            - input_files/ (if save_inputs)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_mesh.export(output_dir / "mesh.obj")
        self.mesh.export(output_dir / "posed_mesh.obj")

        if save_splat and "gaussian" in self._outputs:
            # Save with transform=None so the PLY aligns with raw_mesh.obj
            # (both in the model's internal Z-up object-centric frame).
            # The default save_ply transform applies a Y-Z swap which would
            # misalign the Gaussian PLY relative to the raw mesh OBJ.
            self._outputs["gaussian"][0].save_ply(
                str(output_dir / "gaussian.ply"), transform=None
            )
            # Also save camera-frame Gaussian PLY aligned with posed_mesh.obj.
            # Build the same compound transform applied to the mesh:
            #   pose_matrix → undo cam2ncam (translate then scale).
            cam2ncam_inv_translate = np.eye(4)
            cam2ncam_inv_translate[:3, 3] = -self.cam2ncam[:3, 3]
            cam2ncam_inv_scale = np.eye(4)
            cam2ncam_inv_scale[:3, :3] *= 1.0 / self.cam2ncam[0, 0]
            final_transform = cam2ncam_inv_scale @ cam2ncam_inv_translate @ self.pose_matrix
            self._outputs["gaussian"][0].save_ply(
                str(output_dir / "posed_gaussian.ply"), transform=final_transform
            )

        if save_glb:
            from recgen_inference.recgen_modules.utils import postprocessing_utils

            if getattr(postprocessing_utils, "_HAS_GLB_DEPS", False):
                glb = postprocessing_utils.to_glb(
                    self._outputs["gaussian"][0],
                    self._outputs["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                glb.export(str(output_dir / "textured_mesh.glb"))
            else:
                print(
                    "[recgen_inference] textured_mesh.glb skipped: nvdiffrast / "
                    "diff_gaussian_rasterization / xatlas / pymeshfix not installed. "
                    "Install with `pixi run build-nvdiffrast` and "
                    "`pixi run build-gaussian-rasterizer`."
                )

        overlay = _render_overlay(self.mesh, self.rgb, self.intrinsics)
        Image.fromarray(overlay).save(output_dir / "overlay.png")

        if "gaussian" in self._outputs and "mesh" in self._outputs:
            missing = _missing_turntable_deps()
            if missing:
                cmds = {
                    "nvdiffrast": "pixi run build-nvdiffrast",
                    "diff_gaussian_rasterization": "pixi run build-gaussian-rasterizer",
                }
                hint = "; ".join(f"`{cmds[m]}`" for m in missing)
                print(
                    f"[recgen_inference] turntable.mp4 skipped: missing {', '.join(missing)}. "
                    f"Install with {hint}."
                )
            else:
                try:
                    _render_turntable(
                        self._outputs["gaussian"][0],
                        self._outputs["mesh"][0],
                        output_dir / "turntable.mp4",
                    )
                except Exception as e:
                    print(
                        f"[recgen_inference] turntable.mp4 failed during rendering: "
                        f"{type(e).__name__}: {e}"
                    )

        metadata: Dict[str, Any] = {
            "pose_matrix": self.pose_matrix.tolist(),
            "pose_quat": self.pose_quat.tolist(),
            "pose": {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating,)) else v
                for k, v in self.pose.items()
            },
            "cam2ncam": self.cam2ncam.tolist(),
            "pose_representation": self.pose_representation,
            "intrinsics": np.asarray(self.intrinsics).tolist(),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if self.posthoc_color is not None:
            with open(output_dir / "posthoc_color.json", "w") as f:
                json.dump(self.posthoc_color, f)

        if save_inputs:
            inp_dir = output_dir / "input_files"
            inp_dir.mkdir(exist_ok=True)
            np.save(inp_dir / "cam2ncam.npy", self.cam2ncam)
            np.save(inp_dir / "intrinsics.npy", np.asarray(self.intrinsics))
            Image.fromarray(self.rgb).save(inp_dir / "rgb.png")


def build_result(
    mesh: trimesh.Trimesh,
    raw_mesh: trimesh.Trimesh,
    pose_matrix: np.ndarray,
    parsed_pose: Dict[str, Any],
    cam2ncam: np.ndarray,
    pose_representation: str,
    outputs: Dict[str, Any],
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    posthoc_color: Optional[Dict[str, Any]] = None,
) -> RecGenResult:
    """Construct a RecGenResult, deriving pose_quat from the parsed components."""
    trans = np.asarray(parsed_pose["translation"], dtype=np.float64)
    quat = np.asarray(parsed_pose["quaternion"], dtype=np.float64)
    pose_quat = np.concatenate([trans, quat])
    return RecGenResult(
        mesh=mesh,
        raw_mesh=raw_mesh,
        pose_matrix=pose_matrix,
        pose_quat=pose_quat,
        pose=parsed_pose,
        cam2ncam=cam2ncam,
        pose_representation=pose_representation,
        rgb=np.asarray(rgb),
        intrinsics=np.asarray(intrinsics),
        _outputs=outputs,
        posthoc_color=posthoc_color,
    )


def _render_overlay(
    mesh: trimesh.Trimesh,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    mesh_color: tuple = (116, 92, 180),
    opacity: float = 0.6,
) -> np.ndarray:
    """Project the camera-frame mesh into `rgb` with painters' algorithm + Lambert shading."""
    import cv2

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    h, w = rgb.shape[:2]
    K = np.asarray(intrinsics, dtype=np.float64)

    z = verts[:, 2]
    valid = z > 1e-3
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * verts[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * verts[:, 1] / z_safe + K[1, 2]
    pixels = np.stack([u, v], axis=1)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(nlen > 0, nlen, 1.0)
    diffuse = np.clip(0.25 + 0.75 * np.abs(normals[:, 2]), 0, 1)

    face_z = z[faces].mean(axis=1)
    face_valid = valid[faces].all(axis=1)
    order = np.argsort(-face_z)

    mc = np.array(mesh_color, dtype=np.float32)
    rendered = np.zeros((h, w, 3), dtype=np.uint8)
    mask_img = np.zeros((h, w), dtype=np.uint8)
    for fi in order:
        if not face_valid[fi]:
            continue
        pts = pixels[faces[fi]].astype(np.int32)
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
            continue
        color = (mc * diffuse[fi]).astype(np.uint8).tolist()
        cv2.fillPoly(rendered, [pts.reshape(-1, 1, 2)], color)
        cv2.fillPoly(mask_img, [pts.reshape(-1, 1, 2)], 255)

    bg = rgb.astype(np.float32) / 255.0
    mf = (mask_img > 0).astype(np.float32)[:, :, None]
    mesh_f = rendered.astype(np.float32) / 255.0
    comp = bg * (1 - opacity * mf) + mesh_f * opacity * mf
    return np.clip(comp * 255, 0, 255).astype(np.uint8)


def _missing_turntable_deps() -> list:
    """Return the list of missing CUDA extensions required for turntable.mp4."""
    import importlib.util
    missing = []
    for name in ("nvdiffrast", "diff_gaussian_rasterization"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def _render_turntable(
    gaussian: Any,
    mesh_extract: Any,
    output_path: Path,
    num_frames: int = 120,
    fps: int = 30,
) -> None:
    """Render a side-by-side turntable: gaussian color | mesh normals."""
    import imageio.v2 as imageio
    from recgen_inference.recgen_modules.utils import render_utils

    color = render_utils.render_video(gaussian, num_frames=num_frames)["color"]
    normal = render_utils.render_video(mesh_extract, num_frames=num_frames)["normal"]
    frames = [np.concatenate([color[i], normal[i]], axis=1) for i in range(len(color))]
    imageio.mimsave(str(output_path), frames, fps=fps)
