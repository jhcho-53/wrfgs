# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WRF-GS+ (TWC 2025) reconstructs the **wireless radiation field (WRF)** for channel modeling using **3D Gaussian Splatting**. It is a fork of INRIA's [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) where the analogy is repurposed: instead of learning RGB appearance from camera photos, it learns to synthesize the **spatial spectrum** (a 90×360 elevation×azimuth power map at a fixed receiver/gateway antenna) as a function of the **transmitter position**. The paper is "Neural Representation for Wireless Radiation Field Reconstruction" (IEEE TWC 2025, doi 10.1109/TWC.2025.3631663).

There are now **two pipelines** in this repo:

- **(A) Original per-scene WRF-GS+** — [train.py](train.py). One model memorizes ONE environment, conditioned only on `tx_pos`, with no scene-geometry input. This is the published method.
- **(B) Scene-conditioned / cross-scene WRF-GS+** (active work on branch `wrfgs-multimodal-adaptation`) — [train_multiscene.py](train_multiscene.py). Gaussians are **frozen on each scene's static RSU-lidar geometry** and a **single shared MLP**, conditioned on **relative single-bounce geometry**, predicts the per-Gaussian EM signal. Trained jointly over many scenes; a held-out scene generalizes **zero-shot** by plugging in its lidar. This targets the CARLA+Sionna V2I **Multimodal-Wireless** dataset (arXiv:2511.03220) and is the subject of the research log below.

**Before working on pipeline B, read [docs/research/2026-06-18-scene-conditioned-wrf-generalization.md](docs/research/2026-06-18-scene-conditioned-wrf-generalization.md)** — it is the authoritative record of the method, the key findings (absolute vs relative conditioning, why SSIM is the wrong metric, the scene-count→generalization result), the literature positioning, and the B200 port. Design spec and plan: [docs/superpowers/](docs/superpowers/).

### The 3DGS reinterpretation (shared by both pipelines)
- **"Camera"** = the fixed gateway / RSU (RX antenna). Its pose comes from `gateway_info.yml` (`position` → `r_o`, `orientation` quaternion → rotation `R`), built per-iteration in [utils/generate_camera.py](utils/generate_camera.py) as a 180°-FoV, 360×90 virtual camera.
- **"Time" / deformation conditioning** = the transmitter position `tx_pos` (3D), one per spectrum sample.
- **Rendered output** = a 2-channel image read as **real + imaginary** parts of the spectrum; the predicted magnitude is `|real + i·imag|`, compared against the ground-truth spectrum PNG with L1 + (fused/skimage) SSIM loss.
- **`d_xyz` is intentionally NOT applied** by the rasterizer in either pipeline ([gaussian_renderer/__init__.py:59-60](gaussian_renderer/__init__.py#L59) — `means3D = pc.get_xyz`; the `+ d_xyz` line is commented). The deform MLP still outputs `d_xyz`; it's available to switch on ("A+deform" capacity escape hatch) but currently unused.

## Commands

```bash
# Environment. The repo environment.yml targets the original stack
# (Python 3.8, PyTorch 1.13.1, CUDA 11.6) and does NOT build on Blackwell (B200/sm_100).
# The active working env is the B200 port: conda `wrfgsplus` (py3.11 + torch 2.12 cu130),
# with the rasterizer forced to -gencode arch=compute_100,code=sm_100 (see research log §7).
conda env create --file environment.yml   # original stack only
conda activate wrfgsplus

# Build the three CUDA extension submodules (must succeed before training).
# Editing any .cu requires reinstalling that submodule.
cd submodules
pip install ./simple-knn
pip install ./diff-gaussian-rasterization
pip install ./fused-ssim
cd ..

# (A) Per-scene training. --datadir selects the dataset (default ./data_test200).
python train.py
python train.py --datadir data_mw_Town05_parkinglot --gpu 1 --disable_viewer
python train.py --start_checkpoint output/<timestamp>/chkpnt30000.pth

# (B) Scene-conditioned / cross-scene training. Each --scenes / --holdout-scenes
# entry is "DATADIR:LIDAR_NPY" (LIDAR_NPY from tools/extract_rsu_scene.py, written to scenes_static/).
# Held-out scenes are eval-only (zero-shot, never trained). Iterations come from --iterations.
python train_multiscene.py \
  --scenes data_mw_Town05_parkinglot:scenes_static/Town05_parkinglot.npy \
           data_mw_Town05_CBDcrossroad:scenes_static/Town05_CBDcrossroad.npy \
  --holdout-scenes data_mw_Town05_ringroad:scenes_static/Town05_ringroad.npy \
  --iterations 12000 --eval-every 4000

# Data generation (Multimodal-Wireless V2I -> WRF-GS+ format). Run per scenario:
python -m tools.mw2wrfgs.cli --town Town05 --scenario Town05_parkinglot --out-dir data_mw_Town05_parkinglot
python tools/extract_rsu_scene.py --town Town05 --scenario Town05_parkinglot --out scenes_static/Town05_parkinglot.npy

# Tests (cover only the tools/mw2wrfgs converter; the training pipeline has none)
pytest tests/
```

There is **no linter or CI**, and the training code has **no test suite**. "Verification" of training means inspecting metrics and rendered-vs-GT comparison PNGs:
- Pipeline A: SSIM / mean-pixel-error and comparison PNGs under `logs/<timestamp>/`; TensorBoard + checkpoints (`.pth`) / point clouds (`.ply`) under `output/<timestamp>/`.
- Pipeline B: printed per-scene and **zero-shot held-out** `{ssim, az_err_deg, el_err_deg}` every `--eval-every` steps. **SSIM saturates on these smooth targets — the power-weighted DoA `az_err_deg` is the discriminative metric** (see research log §4.3).

`pytest tests/` exercises `tools/mw2wrfgs` (geometry, spectrum synthesis, dataio, assemble) and the `--datadir` plumbing; [tests/conftest.py](tests/conftest.py) puts both repo root and `tools/` on the path.

## Architecture

### Pipeline A — per-scene (train.py)
Training loop lives entirely in [train.py](train.py) `training()`. Each iteration:
1. Pull `(spectrum, tx_pos)` from the dataset (`scene.train_iter_dataset`).
2. Build the virtual camera from the fixed gateway pose (`scene.r_o`, `scene.gateway_orientation`).
3. Run the **deformation MLP**: `deform.step(gaussians.get_xyz, tx_pos.expand(N,-1))` → `d_xyz, d_rotation, d_scaling, d_signal`.
4. **Render** via the custom rasterizer, applying the deltas (except `d_xyz`).
5. Convert the 2-channel render to a magnitude spectrum and backprop L1 + SSIM against the GT spectrum.
6. Standard 3DGS **densification / pruning / opacity reset** runs until `densify_until_iter`.

### Pipeline B — scene-conditioned / cross-scene (train_multiscene.py)
[train_multiscene.py](train_multiscene.py) holds the whole loop. Key differences from A:
- **No densification, no per-scene parameters.** Each scene is a `SceneHolder`: a fixed camera (its gateway pose) + Gaussians **frozen** on its lidar via `GaussianModel.init_from_lidar` (positions = lidar points, SH = 0, every per-Gaussian attribute `requires_grad=False`). The scene enters training **only** through its lidar geometry.
- **One SHARED [RelDeformModel](utils/rel_deform.py)** is the only thing learned. Scenes are mixed round-robin within/across steps; a held-out scene is evaluated zero-shot.
- **`tx_to_cam`**: the stored `tx_pos` is normalized (RSU-local, `R_rsu⁻¹(cav−rsu)/scale`); `SceneHolder` reconstructs it back to Rx-frame metres using the per-scene `scale`+`yaw` cached in `<datadir>/meta.json` (computed by `scene_scale_yaw`, which calls back into `tools/mw2wrfgs`). The world-z-up→camera-y-up axis permutation `[x,z,y]` (`to_cam_frame`) keeps lidar Gaussians and the target spectrum in one frame.
- **`angular_err`** is the power-weighted circular-mean DoA azimuth/elevation error used by `evaluate()`.

### Modules
- [scene/__init__.py](scene/__init__.py) — `Scene` wires the dataset + gateway pose for **pipeline A**. `self.datadir = args.datadir` (now a real `--datadir` flag, default `./data_test200`); `cameras_extent = 2`, `batch_size = 1` are still hardcoded here.
- [scene/dataloader.py](scene/dataloader.py) — `Spectrum_dataset` ("rfid" type). Loads `spectrum/NNNNN.png` (÷255) and the matching row of `tx_pos.csv` (1-indexed by filename). `split_dataset()` writes `train_index.txt` / `test_index.txt` if absent. Shared by both pipelines.
- [scene/gaussian_model.py](scene/gaussian_model.py) — `GaussianModel`. **Two init paths:** `gaussian_init()` (pipeline A) seeds **200k random Gaussians** (`randn*20`); `init_from_lidar()` (pipeline B) places **frozen** Gaussians on lidar points. The COLMAP `create_from_pcd()` is dead (see below). Densify/prune/SH/optimizer are stock 3DGS.
- [scene/deform_model.py](scene/deform_model.py) + [utils/pos_utils.py](utils/pos_utils.py) — `DeformModel` / `DeformNetwork`: the **absolute-coordinate** 8-layer MLP with positional encoding on Gaussian xyz **and** `tx_pos`. Outputs `d_xyz`, `d_rotation`, `d_scaling`, and a **complex signal** delta (`signal_real · exp(i·signal_phase)` → magnitude). Used by pipeline A; the cross-scene baseline that **memorizes and fails to generalize**.
- [utils/rel_deform.py](utils/rel_deform.py) — `RelDeformModel` / `RelDeformNetwork`, the **pipeline-B contribution**. Same MLP shape and outputs, but conditioned on per-scatterer **relative single-bounce geometry** `rel_geom_features(p, tx)` = `[d_rx(3), d_tx(3), bisector(3), cos(bounce), log r_rx, log r_tx, log path]` (13-D) instead of absolute coords. This is what turns memorization into cross-scene generalization (research log §3.3, §5.1).
- [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py) — `render()`. Adds `d_scaling`/`d_rotation` to scaling/rotation and `d_signal` to SH features; **`d_xyz` not applied** (`means3D = pc.get_xyz`). `network_gui.py` is the live SplatViz viewer (default port 6074; `--disable_viewer`). Note the B200 port added a defensive non-finite-covariance cull for the equirectangular pole singularity.
- [arguments/__init__.py](arguments/__init__.py) — stock 3DGS `ModelParams`/`PipelineParams`/`OptimizationParams`; the real training hyperparameters live here. `--datadir` is the one wireless-specific addition (`ModelParams.datadir`).
- `submodules/` — three CUDA extensions: `diff-gaussian-rasterization` (custom; antialiasing / separate-SH / `SparseGaussianAdam`), `simple-knn`, `fused-ssim`.

### Data-generation toolchain (`tools/`)
Converts the raw Multimodal-Wireless V2I drop (CARLA scene + Sionna ray-traced CSI, default root `/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless`) into the `data_test200` on-disk format.
- [tools/mw2wrfgs/](tools/mw2wrfgs/) — the converter package (run as `python -m tools.mw2wrfgs.cli`):
  - `geometry.py` — `angles_to_pixel` (Sionna world-frame arrival angles → equirectangular renderer pixels) and `build_tx_pos` (`R_rsu⁻¹(cav−rsu)/scale`, RSU-local conditioning coords).
  - `spectrum.py` — per-Sionna-path power → broad anisotropic Gaussian lobes over a floor (densified, per-image peak-normalized) → uint8 PNG.
  - `dataio.py` / `assemble.py` — seed-dir resolution, sample enumeration, scale measurement, dataset + 80/20 split + identity-gauge `gateway_info.yml` writer. `cli.py` asserts a cull-radius margin (`min ||tx_pos||`).
- [tools/extract_rsu_scene.py](tools/extract_rsu_scene.py) — aggregates ~20 RSU lidar frames and keeps **voxel-occupancy-static** points (the channel is static; moving cars are dropped), voxel-downsamples, writes the `(N,3)` `.npy` scene cloud used by `init_from_lidar`.
- `calibrate_spectrum.py`, `check_alignment.py`, `respit_spatial.py`, `qualitative_figure.py` — calibration / GPU-alignment / figure-generation helpers.

### Inherited-but-unused 3DGS code (don't be misled)
Because this is a 3DGS fork, much code assumes photographic, COLMAP-reconstructed scenes and is **dead for the wireless pipelines**: [convert.py](convert.py) (COLMAP runner), [scene/colmap_loader.py](scene/colmap_loader.py), [scene/dataset_readers.py](scene/dataset_readers.py), `create_from_pcd()`, the `lpipsPyTorch/` LPIPS metric, and CLI flags like `--source_path` / `--images` / `--eval`. The active data paths are `Spectrum_dataset` + (`gaussian_init()` for A / `init_from_lidar()` for B).

## Data layout

Every converted dataset (`data_test200/`, each `data_mw_<Town>_<scenario>/`) follows one format:
- `spectrum/NNNNN.png` — spatial-spectrum images (90×360), the training/eval targets.
- `tx_pos.csv` — transmitter `x,y,z`; row `N` corresponds to `NNNNN.png` (1-indexed). For `data_mw_*` these are RSU-local **normalized** coords.
- `gateway_info.yml` — receiver/RSU `position` and `orientation` (quaternion). For `data_mw_*` it is an **identity gauge**.
- `train_index.txt` / `test_index.txt` — 80/20 split of sample IDs (auto-generated if absent).
- `meta.json` (`data_mw_*` only) — cached `{scale, yaw}` needed to de-normalize `tx_pos` back to Rx-frame metres in pipeline B.

Other top-level data:
- `data_test200/` — the bundled 200-sample sanity dataset (sourced from [NeRF2](https://github.com/XPengZhao/NeRF2)); pipeline A default.
- `data_mw_<Town>_<scenario>/` — the 16 converted CARLA/Sionna V2I scenarios across Town03/05/07/10 (`.gitignore`d — regenerate with the converter).
- `scenes_static/<Town>_<scenario>.npy` — frozen static-scene lidar clouds for pipeline B (`.gitignore`d — regenerate with `extract_rsu_scene.py`).
