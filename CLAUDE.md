# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WRF-GS+ (TWC 2025) reconstructs the **wireless radiation field (WRF)** for channel modeling using **3D Gaussian Splatting**. It is a fork of INRIA's [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) where the analogy is repurposed: instead of learning RGB appearance from camera photos, it learns to synthesize the **spatial spectrum** (a 90×360 elevation×azimuth power map at a fixed receiver/gateway antenna) as a function of the **transmitter position**. The paper is "Neural Representation for Wireless Radiation Field Reconstruction" (IEEE TWC 2025, doi 10.1109/TWC.2025.3631663).

The key reinterpretation of the 3DGS pipeline:
- **"Camera"** = the fixed gateway (RX antenna). Its pose comes from `gateway_info.yml` (`position` → `r_o`, `orientation` quaternion → rotation `R`), built per-iteration in [utils/generate_camera.py](utils/generate_camera.py) as a 180°-FoV, 360×90 virtual camera.
- **"Time" / deformation conditioning** = the transmitter position `tx_pos` (3D), one per spectrum sample.
- **Rendered output** = a 2-channel image read as **real + imaginary** parts of the spectrum; the predicted magnitude is `|real + i·imag|`, compared against the ground-truth spectrum PNG with L1 + (fused/skimage) SSIM loss.

## Commands

```bash
# Environment (CUDA GPU required: Python 3.8, PyTorch 1.13.1, CUDA 11.6)
conda env create --file environment.yml
conda activate wrfgsplus

# Build the three CUDA extension submodules (must succeed before training)
cd submodules
pip install ./simple-knn
pip install ./diff-gaussian-rasterization
pip install ./fused-ssim
cd ..

# Train (also runs periodic evaluation). Defaults to the bundled data_test200 dataset.
python train.py
python train.py --gpu 1 --iterations 200000 --disable_viewer
python train.py --start_checkpoint output/<timestamp>/chkpnt30000.pth
```

There is **no test suite, linter, or CI**. "Verification" here means running training and inspecting the SSIM / mean-pixel-error metrics and rendered-vs-GT comparison PNGs written under `logs/<timestamp>/`. Outputs and TensorBoard logs go to `output/<timestamp>/`; checkpoints (`.pth`) and Gaussian point clouds (`.ply`) are saved at the `--save_iterations` / `--checkpoint_iterations` steps.

## Architecture

Training loop lives entirely in [train.py](train.py) `training()`. Each iteration:
1. Pull `(spectrum, tx_pos)` from the dataset (`scene.train_iter_dataset`).
2. Build the virtual camera from the fixed gateway pose (`scene.r_o`, `scene.gateway_orientation`).
3. Run the **deformation MLP**: `deform.step(gaussians.get_xyz, tx_pos.expand(N,-1))` → `d_xyz, d_rotation, d_scaling, d_signal`.
4. **Render** via the custom rasterizer, applying the deltas.
5. Convert the 2-channel render to a magnitude spectrum and backprop L1 + SSIM against the GT spectrum.
6. Standard 3DGS **densification / pruning / opacity reset** runs until `densify_until_iter`.

Modules:
- [scene/__init__.py](scene/__init__.py) — `Scene` wires the dataset and gateway pose. **Many knobs are hardcoded here, not CLI args**: `self.datadir = "./data_test200"`, `cameras_extent = 2`, `batch_size = 1`. Change the dataset by editing `datadir`.
- [scene/dataloader.py](scene/dataloader.py) — `Spectrum_dataset` ("rfid" type). Loads `spectrum/NNNNN.png` (÷255) and the matching row of `tx_pos.csv` (1-indexed by filename). `split_dataset()` writes `train_index.txt` / `test_index.txt` if absent.
- [scene/gaussian_model.py](scene/gaussian_model.py) — `GaussianModel`. Note `gaussian_init()` initializes **200k Gaussians randomly** (`randn*20`), bypassing the COLMAP-based `create_from_pcd()`. The rest (densify/prune/SH/optimizer) is stock 3DGS.
- [scene/deform_model.py](scene/deform_model.py) + [utils/pos_utils.py](utils/pos_utils.py) — `DeformModel` wraps `DeformNetwork`, an 8-layer MLP with positional encoding on both Gaussian xyz and `tx_pos`. Outputs per-Gaussian `d_xyz`, `d_rotation`, `d_scaling`, and a **complex signal** delta (`signal_real · exp(i·signal_phase)` → magnitude) added to SH features. This is the WRF-GS+ "deformable Gaussian" physics extension.
- [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py) — `render()`. Adds `d_scaling`/`d_rotation` to scaling/rotation and `d_signal` to SH features. **`d_xyz` is intentionally NOT applied** (`means3D = pc.get_xyz`; the `+ d_xyz` line is commented out). `network_gui.py` is the live SplatViz viewer (default port 6074; disable with `--disable_viewer`).
- [arguments/__init__.py](arguments/__init__.py) — stock 3DGS `ModelParams`/`PipelineParams`/`OptimizationParams`. The real training hyperparameters (iterations, LR schedules, densification) come from here.
- `submodules/` — three CUDA extensions: `diff-gaussian-rasterization` (custom, supports antialiasing / separate-SH / `SparseGaussianAdam`), `simple-knn`, `fused-ssim`. Editing `.cu` files requires reinstalling the submodule.

### Inherited-but-unused 3DGS code (don't be misled)

Because this is a 3DGS fork, much code assumes photographic, COLMAP-reconstructed scenes and is **dead for the wireless pipeline**: [convert.py](convert.py) (COLMAP runner), [scene/colmap_loader.py](scene/colmap_loader.py), [scene/dataset_readers.py](scene/dataset_readers.py), `create_from_pcd()`, the `lpipsPyTorch/` LPIPS metric, and CLI flags like `--source_path` / `--images` / `--eval`. The active data path is the `Spectrum_dataset` + `gaussian_init()` route described above.

## Data layout (`data_test200/`)

- `spectrum/NNNNN.png` — 200 spatial-spectrum images (90×360), the training/eval targets.
- `tx_pos.csv` — transmitter `x,y,z` positions; row `N` corresponds to `NNNNN.png` (1-indexed).
- `gateway_info.yml` — receiver antenna `position` and `orientation` (quaternion).
- `train_index.txt` / `test_index.txt` — auto-generated 80/20 split of sample IDs.

More datasets follow the same format and come from the [NeRF2](https://github.com/XPengZhao/NeRF2) repo.
