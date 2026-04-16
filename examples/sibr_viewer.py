"""SIBR-compatible viewer for 3D Gaussian Splatting.

Supports single view and side-by-side dual view for comparing two models.

Usage:
    # Single view
    python sibr_viewer.py -m /path/to/model

    # Dual view (side-by-side comparison)
    python sibr_viewer.py -m /path/to/model_A /path/to/model_B

    # With COLMAP source data
    python sibr_viewer.py -m /path/to/model_A /path/to/model_B -s /path/to/colmap
"""

import argparse
import json
import math
import os
import time
from typing import Tuple

import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import viser

from gsplat.rendering import rasterization
from utils import rotation_6d_to_matrix


def load_ply(ply_path: str, device: torch.device):
    """Load a standard 3DGS PLY file and return gsplat-ready tensors.

    Returns:
        means [N, 3], quats [N, 4], scales [N, 3], opacities [N],
        colors [N, K, 3], sh_degree (int)
    """
    with open(ply_path, "rb") as f:
        # Parse ASCII header
        properties = []
        vertex_count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        # Build numpy structured dtype from properties
        dtype = np.dtype([(name, "f4") for name in properties])
        data = np.frombuffer(
            f.read(vertex_count * dtype.itemsize), dtype=dtype
        ).copy()

    # Extract fields
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
    opacities_raw = data["opacity"]
    scales_raw = np.stack(
        [data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1
    )
    rots = np.stack(
        [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1
    )

    # SH DC coefficients
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1)

    # SH rest coefficients — count how many f_rest_* fields exist
    f_rest_names = [p for p in properties if p.startswith("f_rest_")]
    n_rest = len(f_rest_names)

    if n_rest > 0:
        f_rest_flat = np.stack(
            [data[name] for name in f_rest_names], axis=-1
        )  # [N, n_rest]
        n_rest_per_channel = n_rest // 3
        # PLY stores channel-major: f_rest_0..K = ch0, f_rest_K+1..2K = ch1, ...
        f_rest = f_rest_flat.reshape(vertex_count, 3, n_rest_per_channel).transpose(
            0, 2, 1
        )  # [N, n_rest_per_channel, 3]
        sh_degree = int(math.sqrt(n_rest_per_channel + 1)) - 1
    else:
        f_rest = np.zeros((vertex_count, 0, 3), dtype=np.float32)
        sh_degree = 0

    # Combine SH: [N, 1, 3] + [N, K-1, 3] → [N, K, 3]
    f_dc = f_dc.reshape(vertex_count, 1, 3)
    colors = np.concatenate([f_dc, f_rest], axis=1)

    # Convert to tensors with activations
    means = torch.from_numpy(xyz).float().to(device)
    quats = torch.from_numpy(rots).float().to(device)
    scales = torch.exp(torch.from_numpy(scales_raw).float().to(device))
    opacities = torch.sigmoid(
        torch.from_numpy(opacities_raw).float().to(device)
    )
    colors = torch.from_numpy(colors).float().to(device)

    return means, quats, scales, opacities, colors, sh_degree


def load_colmap_cameras(source_path: str):
    """Load camera parameters from COLMAP sparse reconstruction."""
    from pycolmap import SceneManager

    colmap_dir = os.path.join(source_path, "sparse", "0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(source_path, "sparse")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    imdata = manager.images
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    w2c_mats = []
    first_camera_id = None

    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
        if first_camera_id is None:
            first_camera_id = im.camera_id

    w2c_mats = np.stack(w2c_mats, axis=0)
    camtoworlds = np.linalg.inv(w2c_mats)

    cam = manager.cameras[first_camera_id]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    width, height = cam.width, cam.height

    return camtoworlds, K, width, height


def load_cameras_json(json_path: str):
    """Load camera parameters from cameras.json (standard 3DGS format)."""
    with open(json_path) as f:
        cameras = json.load(f)

    camtoworlds = []
    for cam in cameras:
        R = np.array(cam["rotation"])
        pos = np.array(cam["position"])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = pos
        camtoworlds.append(c2w)

    camtoworlds = np.stack(camtoworlds, axis=0)

    first = cameras[0]
    fx, fy = first["fx"], first["fy"]
    width, height = first["width"], first["height"]
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return camtoworlds, K, width, height


def compute_alignment(c2ws_a: np.ndarray, c2ws_b: np.ndarray) -> np.ndarray:
    """Compute similarity transform from coordinate system A to B.

    Uses Procrustes analysis on corresponding camera positions.
    Returns a 4x4 matrix T such that a point in A's frame maps to B's frame:
        p_B = T @ p_A
    """
    pts_a = c2ws_a[:, :3, 3]  # [N, 3]
    pts_b = c2ws_b[:, :3, 3]  # [N, 3]

    centroid_a = pts_a.mean(axis=0)
    centroid_b = pts_b.mean(axis=0)

    a_centered = pts_a - centroid_a
    b_centered = pts_b - centroid_b

    # Scale
    scale = np.linalg.norm(b_centered) / np.linalg.norm(a_centered)

    # Rotation via SVD
    H = a_centered.T @ b_centered  # [3, 3]
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, d])  # correct reflection
    R = Vt.T @ S @ U.T

    # Translation
    t = centroid_b - scale * R @ centroid_a

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def _rasterize_rgb(scene, viewmat, K_view, W, H):
    """Render a scene to an RGB image tensor [H, W, 3] with gradients flowing through viewmat."""
    means, quats, scales, opacities, colors, sh_degree = scene
    render_colors, _, _ = rasterization(
        means, quats, scales, opacities, colors,
        viewmat[None], K_view[None], W, H,
        sh_degree=sh_degree,
        render_mode="RGB",
        radius_clip=3,
    )
    return render_colors[0, ..., 0:3]


def refine_alignment_photometric(
    scenes,
    camtoworlds_a: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    T_init: np.ndarray,
    device: torch.device,
    iters: int = 2000,
    downscale: int = 1,
    ssim_lambda: float = 0.2,
    seed: int = 0,
    coarse_to_fine: bool = True,
    early_stop_patience: int = 200,
    early_stop_threshold: float = 1e-5,
) -> np.ndarray:
    """Refine alignment transform T by minimizing photometric loss between A and B renders.

    The residual is parameterized as a Sim(3) delta on top of T_init:
        T = T_init @ delta
        delta = [[exp(ds) * R(drot), dx], [0, 1]]

    Only delta parameters (10 scalars) are optimized. Gaussian parameters are frozen.
    """
    from fused_ssim import fused_ssim

    torch.manual_seed(seed)

    scene_a, scene_b = scenes[0], scenes[1]

    T_init_t = torch.from_numpy(T_init).float().to(device)
    c2ws_a = torch.from_numpy(camtoworlds_a).float().to(device)
    N = c2ws_a.shape[0]

    # Pre-compute viewmats for model A (never change)
    viewmats_a = torch.inverse(c2ws_a)  # [N, 4, 4]

    identity_6d = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device
    )

    dx = torch.nn.Parameter(torch.zeros(3, device=device))
    drot = torch.nn.Parameter(torch.zeros(6, device=device))
    ds = torch.nn.Parameter(torch.zeros(1, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [dx], "lr": 1e-3},
            {"params": [drot], "lr": 1e-3},
            {"params": [ds], "lr": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iters, eta_min=1e-6
    )

    bottom_row = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32
    )

    def build_T_final():
        R = rotation_6d_to_matrix(drot + identity_6d)  # [3, 3]
        s = torch.exp(ds)  # [1]
        sR = s * R  # [3, 3]
        top = torch.cat([sR, dx.unsqueeze(1)], dim=1)  # [3, 4]
        delta = torch.cat([top, bottom_row], dim=0)  # [4, 4]
        return T_init_t @ delta

    # Build coarse-to-fine resolution stages
    if coarse_to_fine and downscale < 8:
        stages = []
        ds_val = min(8, downscale * 8)
        while ds_val > downscale:
            stages.append(ds_val)
            ds_val = max(downscale, ds_val // 2)
        stages.append(downscale)
        # e.g. downscale=1 -> [8, 4, 2, 1]
    else:
        stages = [downscale]
    iters_per_stage = iters // len(stages)

    print(
        f"  Photometric refinement: {iters} iters, N={N} cameras, "
        f"stages={['1/' + str(s) if s > 1 else 'full' for s in stages]}"
    )
    start_time = time.time()
    loss_ema = None
    initial_loss = None
    global_step = 0

    for stage_idx, stage_ds in enumerate(stages):
        W = max(1, int(round(width / stage_ds)))
        H = max(1, int(round(height / stage_ds)))
        K_scaled = K.copy().astype(np.float32)
        K_scaled[0] *= W / width
        K_scaled[1] *= H / height
        K_t = torch.from_numpy(K_scaled).float().to(device)

        # Pre-cache all model A renders at this resolution
        cache_a = []
        with torch.no_grad():
            for i in range(N):
                img = _rasterize_rgb(scene_a, viewmats_a[i], K_t, W, H)
                cache_a.append(img.clamp(0.0, 1.0))

        if len(stages) > 1:
            print(f"  Stage {stage_idx + 1}/{len(stages)}: {W}x{H} ({iters_per_stage} iters)")

        # Early stopping state (reset per stage)
        best_ema = None
        steps_no_improve = 0

        for step in range(iters_per_stage):
            idx = int(torch.randint(0, N, (1,)).item())
            c2w_a = c2ws_a[idx]

            img_a = cache_a[idx]

            T_final = build_T_final()
            c2w_b = T_final @ c2w_a
            viewmat_b = torch.inverse(c2w_b)
            img_b = _rasterize_rgb(scene_b, viewmat_b, K_t, W, H)
            img_b = img_b.clamp(0.0, 1.0)

            # fused_ssim expects [B, C, H, W]
            a_bchw = img_a.permute(2, 0, 1).unsqueeze(0)
            b_bchw = img_b.permute(2, 0, 1).unsqueeze(0)

            l1 = F.l1_loss(img_b, img_a)
            ssim_val = fused_ssim(b_bchw, a_bchw, padding="valid")
            loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - ssim_val)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val
            if initial_loss is None:
                initial_loss = loss_val

            global_step += 1
            if global_step % 100 == 0 or global_step == 1:
                print(
                    f"    step {global_step:5d}/{iters}  loss={loss_val:.5f}  ema={loss_ema:.5f}"
                )

            # Early stopping check
            if early_stop_patience > 0 and step > 100:
                if best_ema is None or loss_ema < best_ema - early_stop_threshold:
                    best_ema = loss_ema
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
                if steps_no_improve >= early_stop_patience:
                    print(f"    Early stop at step {global_step} (EMA plateau: {loss_ema:.5f})")
                    # Advance scheduler for skipped steps
                    for _ in range(iters_per_stage - step - 1):
                        scheduler.step()
                    break

        del cache_a

    elapsed = time.time() - start_time

    with torch.no_grad():
        T_final = build_T_final().detach().cpu().numpy()

    delta_norm = float(np.linalg.norm(T_final - T_init))
    print(
        f"  Refinement done in {elapsed:.1f}s | "
        f"initial loss={initial_loss:.5f} → ema={loss_ema:.5f} | "
        f"||T_final - T_init||_F={delta_norm:.4f}"
    )
    print(
        f"    dx={dx.detach().cpu().numpy()}  "
        f"scale={float(torch.exp(ds).item()):.6f}"
    )
    return T_final


def setup_initial_camera(server, camtoworlds, K, width, height):
    """Configure viser's initial camera from the first training camera."""
    c2w = camtoworlds[0]

    position = c2w[:3, 3]
    forward = c2w[:3, 2]
    up = -c2w[:3, 1]
    look_at = position + forward

    all_ups = -camtoworlds[:, :3, 1]
    avg_up = np.mean(all_ups, axis=0)
    avg_up = avg_up / np.linalg.norm(avg_up)

    server.scene.set_up_direction(tuple(avg_up))
    server.initial_camera.position = tuple(position)
    server.initial_camera.look_at = tuple(look_at)
    server.initial_camera.up = tuple(avg_up)

    fy = K[1, 1]
    fov = 2.0 * math.atan(height / (2.0 * fy))
    server.initial_camera.fov = fov


def main():
    parser = argparse.ArgumentParser(description="SIBR-compatible 3DGS viewer")
    parser.add_argument(
        "-m", "--model_path", type=str, nargs="+", required=True,
        help="Path(s) to trained model(s). Pass two for side-by-side comparison.",
    )
    parser.add_argument(
        "-s", "--source_path", type=str, default=None,
        help="Path to source data (contains sparse/0/). "
        "If omitted, cameras.json in the first model_path is used.",
    )
    parser.add_argument(
        "--iteration", type=int, default=30_000,
        help="Which iteration to load (default: 30000)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--refine", type=int, default=1,
        help="Photometric refinement of dual-view alignment T (1=on, 0=off).",
    )
    parser.add_argument(
        "--refine_iters", type=int, default=2000,
        help="Number of optimizer steps for photometric refinement.",
    )
    parser.add_argument(
        "--refine_downscale", type=int, default=1,
        help="Downscale factor for refinement render resolution (1=full).",
    )
    parser.add_argument(
        "--refine_no_ctf", action="store_true",
        help="Disable coarse-to-fine progressive resolution for refinement.",
    )
    parser.add_argument(
        "--refine_patience", type=int, default=200,
        help="Early stopping patience (0=disable). Stop if EMA loss plateaus.",
    )
    args = parser.parse_args()

    if len(args.model_path) > 2:
        raise ValueError("At most 2 model paths supported.")

    device = torch.device("cuda")
    dual = len(args.model_path) == 2

    # 1. Load PLY(s)
    scenes = []
    for model_path in args.model_path:
        ply_path = os.path.join(
            model_path, "point_cloud",
            f"iteration_{args.iteration}", "point_cloud.ply",
        )
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY not found: {ply_path}")
        print(f"Loading PLY: {ply_path}")
        scene = load_ply(ply_path, device)
        print(f"  {len(scene[0])} Gaussians, SH degree {scene[5]}")
        scenes.append(scene)

    # 2. Load cameras
    alignment_transform = None
    if args.source_path is not None:
        print(f"Loading COLMAP cameras: {args.source_path}")
        camtoworlds, K, cam_width, cam_height = load_colmap_cameras(args.source_path)
    else:
        json_path = os.path.join(args.model_path[0], "cameras.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                "No camera data found. Provide -s (COLMAP source) or "
                "place cameras.json in model_path."
            )
        print(f"Loading cameras.json: {json_path}")
        camtoworlds, K, cam_width, cam_height = load_cameras_json(json_path)

        # For dual mode, compute alignment from model A's coordinate system to model B's
        if dual:
            json_path_b = os.path.join(args.model_path[1], "cameras.json")
            if os.path.exists(json_path_b):
                print(f"Loading cameras.json (model B): {json_path_b}")
                camtoworlds_b, _, _, _ = load_cameras_json(json_path_b)
                n = min(len(camtoworlds), len(camtoworlds_b))
                alignment_transform = compute_alignment(
                    camtoworlds[:n], camtoworlds_b[:n]
                )
                print(f"  Computed alignment transform ({n} camera pairs)")

                if args.refine:
                    alignment_transform = refine_alignment_photometric(
                        scenes,
                        camtoworlds_a=camtoworlds[:n],
                        K=K,
                        width=cam_width,
                        height=cam_height,
                        T_init=alignment_transform,
                        device=device,
                        iters=args.refine_iters,
                        downscale=args.refine_downscale,
                        coarse_to_fine=not args.refine_no_ctf,
                        early_stop_patience=args.refine_patience,
                    )
            else:
                print("Warning: No cameras.json for model B, skipping alignment.")

    print(f"  {len(camtoworlds)} cameras, resolution {cam_width}x{cam_height}")

    # 3. Create viewer server
    server = viser.ViserServer(port=args.port, verbose=False)

    # 4. Set initial camera
    setup_initial_camera(server, camtoworlds, K, cam_width, cam_height)

    # 5. Render helpers
    @torch.no_grad()
    def render_scene(scene, camera_state, img_wh, transform=None):
        means, quats, scales, opacities, colors, sh_degree = scene
        W, H = img_wh
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        if transform is not None:
            c2w = transform @ c2w
        K_view = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
        viewmat = c2w.inverse()

        render_colors, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat[None], K_view[None], W, H,
            sh_degree=sh_degree,
            render_mode="RGB",
            radius_clip=3,
        )
        return render_colors[0, ..., 0:3].cpu().numpy()

    # Precompute alignment as a GPU tensor for rendering
    align_t = None
    if alignment_transform is not None:
        align_t = torch.from_numpy(alignment_transform).float().to(device)

    def viewer_render_fn(
        camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        W, H = img_wh
        if dual:
            half_w = W // 2
            left = render_scene(scenes[0], camera_state, (half_w, H))
            right = render_scene(scenes[1], camera_state, (half_w, H), transform=align_t)
            return np.hstack([left, right])
        else:
            return render_scene(scenes[0], camera_state, (W, H))

    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    mode_str = "dual" if dual else "single"
    print(f"Viewer ({mode_str}) running at http://localhost:{args.port} ... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()
