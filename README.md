# dual_viewer

A side-by-side 3D Gaussian Splatting viewer built on [gsplat](https://github.com/nerfstudio-project/gsplat). Load two independently trained models and compare them in a synchronized dual view — both views share a single camera, so every mouse/keyboard interaction moves them together.

## Features

- **Dual view comparison** — Two models rendered side-by-side at half width each, composited into a single frame
- **Automatic coordinate alignment** — Procrustes (SVD) initialization + differentiable photometric refinement to align models trained in different coordinate systems
- **SIBR-compatible input** — Reads standard 3DGS PLY files and `cameras.json` directly
- **Web-based** — GPU rasterization on the server, streamed to the browser via [viser](https://github.com/nerfstudio-project/viser)

<!-- TODO: add demo screenshot / GIF -->

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

Open `localhost:8080` in a browser.

## How it works

### Dual view

Each scene is rendered at half the viewport width, then concatenated with `np.hstack`. Because both views share a single viser camera state, synchronization is automatic — no separate servers, iframes, or sync logic needed.

### Coordinate alignment

Two models trained on different COLMAP reconstructions live in different world coordinate systems. To display them from the same viewpoint, the viewer computes a similarity transform from model A's coordinate system to model B's:

1. **Procrustes initialization** — SVD-based alignment using corresponding camera positions from both `cameras.json` files.
2. **Photometric refinement** — A Sim(3) residual (10 DoF: translation + 6D rotation + log-scale) is optimized on top of the Procrustes result, minimizing L1 + SSIM loss between actual rendered images. Gradients flow through gsplat's differentiable rasterization via the view matrix.

The refinement includes three speed optimizations enabled by default:
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
| `-m, --model_path` | (required) | Path(s) to trained model(s). Pass two for dual view. |
| `-s, --source_path` | None | Path to COLMAP sparse data |
| `--iteration` | 30000 | Which training iteration to load |
| `--port` | 8080 | Viewer server port |
| `--refine` | 1 | Photometric refinement (1=on, 0=off) |
| `--refine_iters` | 2000 | Total refinement optimizer steps |
| `--refine_downscale` | 1 | Final render resolution downscale factor |
| `--refine_no_ctf` | false | Disable coarse-to-fine progressive resolution |
| `--refine_patience` | 200 | Early stopping patience (0=disable) |

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
