<div align="center">

# dual_viewer

**Synchronized side-by-side 3D Gaussian Splatting viewer**

Load two independently trained models, share one camera, and compare them in real time.

[![gsplat](https://img.shields.io/badge/backend-gsplat-orange)](https://github.com/nerfstudio-project/gsplat)
[![viser](https://img.shields.io/badge/viewer-viser-blue)](https://github.com/nerfstudio-project/viser)
[![SIBR compatible](https://img.shields.io/badge/input-SIBR%20compatible-green)]()

</div>

---

A side-by-side 3D Gaussian Splatting viewer built on [gsplat](https://github.com/nerfstudio-project/gsplat). Load two independently trained models and compare them in a synchronized dual view — both views share a single camera, so every mouse/keyboard interaction moves them together.

## Alignment in action

Two models reconstructed from COLMAP at **different resolutions** end up in different world coordinate systems. The viewer brings them into a shared frame in two stages — a Procrustes initialization, then a photometric refinement on top. The clips below show why both stages matter.

> In every clip, **left = COLMAP run at 1/8 resolution**, **right = COLMAP run at 1/2 resolution**.

<table>
  <tr>
    <td width="33%" align="center"><b>1. Before alignment</b></td>
    <td width="33%" align="center"><b>2. Coarse align</b></td>
    <td width="33%" align="center"><b>3. Fine align</b></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/1c48c244-b642-40de-9962-23097f5f2d27" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/bc954a9e-59e8-40c4-8902-adc819f8b14e" autoplay loop muted playsinline width="100%"></video></td>
    <td><video src="https://github.com/user-attachments/assets/17212482-7096-4eb6-b8a8-3fd00e05f7d9" autoplay loop muted playsinline width="100%"></video></td>
  </tr>
</table>

Models in their original coordinate systems drift apart (1). Procrustes recovers the global pose but leaves residual scale/rotation error (2). The photometric refinement closes the gap so the two track together pixel-for-pixel (3).

## Features

- **Dual view comparison** — Two models rendered side-by-side at half width each, composited into a single frame.
- **Automatic coordinate alignment** — Procrustes (SVD) initialization plus differentiable photometric refinement to align models trained in different coordinate systems.
- **SIBR-compatible input** — Reads standard 3DGS PLY files and `cameras.json` directly.
- **Web-based** — GPU rasterization on the server, streamed to the browser via [viser](https://github.com/nerfstudio-project/viser).

## Installation

```bash
# gsplat (local build)
pip install -e .

# Additional dependencies
pip install viser nerfview pycolmap fused-ssim
```

## Usage

```bash
cd examples

# Single view
python sibr_viewer.py -m /path/to/model

# Dual view (side-by-side)
python sibr_viewer.py -m /path/to/model_A /path/to/model_B

# Specify COLMAP source directly
python sibr_viewer.py -m /path/to/model_A /path/to/model_B -s /path/to/colmap

# Disable photometric refinement
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine 0

# Adjust refinement iterations / resolution
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine_iters 1000 --refine_downscale 2
```

Then open `localhost:8080` in a browser.

## How it works

### Dual view

Each scene is rendered at half the viewport width, then concatenated with `np.hstack`. Because both views share a single viser camera state, synchronization is automatic — no separate servers, iframes, or sync logic needed.

### Coordinate alignment

To display two models from the same viewpoint, the viewer computes a similarity transform from model A's coordinate system to model B's:

1. **Procrustes initialization** — SVD-based alignment using corresponding camera positions from both `cameras.json` files.
2. **Photometric refinement** — A Sim(3) residual (10 DoF: translation + 6D rotation + log-scale) is optimized on top of the Procrustes result, minimizing an L1 + SSIM loss between the actual rendered images. Gradients flow through gsplat's differentiable rasterization via the view matrix.

The refinement includes three speed optimizations, enabled by default:

- **Model A pre-caching** — Model A renders are view-independent targets, so all N camera views are rendered once and cached before the optimization loop.
- **Coarse-to-fine resolution** — Optimization starts at low resolution (1/8) and progressively increases, so early iterations are cheap.
- **Early stopping** — Stops when the EMA loss plateaus, avoiding unnecessary iterations.

## Input format

```
model_path/                          # -m flag
├── cameras.json                     # Camera parameters (used when no COLMAP source)
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply          # Standard 3DGS PLY

source_path/                         # -s flag (optional)
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

Camera loading priority: `-s` COLMAP sparse > `model_path/cameras.json`

## CLI options

| Option | Default | Description |
|---|---|---|
| `-m, --model_path` | *(required)* | Path(s) to trained model(s). Pass two for dual view. |
| `-s, --source_path` | `None` | Path to COLMAP sparse data. |
| `--iteration` | `30000` | Which training iteration to load. |
| `--port` | `8080` | Viewer server port. |
| `--refine` | `1` | Photometric refinement (`1`=on, `0`=off). |
| `--refine_iters` | `2000` | Total refinement optimizer steps. |
| `--refine_downscale` | `1` | Final render resolution downscale factor. |
| `--refine_no_ctf` | `false` | Disable coarse-to-fine progressive resolution. |
| `--refine_patience` | `200` | Early stopping patience (`0`=disable). |

## Acknowledgements

This project uses [gsplat](https://github.com/nerfstudio-project/gsplat) as the CUDA rasterization backend.

```bibtex
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting},
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    journal={arXiv preprint arXiv:2409.06765},
}
```
