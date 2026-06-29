# Building & running WRF-GS+ on NVIDIA B200 (sm_100 / CUDA 13)

The repo's `environment.yml` (Python 3.8, torch 1.13.1, CUDA 11.6) **does not work on Blackwell B200** — sm_100 needs CUDA ≥ 12.8. Use the modern stack below. All source fixes needed for the B200 build are already committed (see *Source fixes* at the bottom); this doc is the recipe to recreate the environment and reproduce the experiment after a fresh `git clone`.

Verified on: 2× NVIDIA B200 (sm_100, compute capability 10.0), driver 580.95.05 (CUDA 13.0), system `nvcc` 13.1, gcc 13.3.

---

## 1. Conda environment

```bash
conda create -n wrfgsplus python=3.11 -y
PY=~/miniconda3/envs/wrfgsplus/bin/python      # use the full path (see gotcha below)

# torch built for CUDA 13 (Blackwell-capable). cu130 matches the system nvcc 13.x.
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu130

# runtime / training deps
$PY -m pip install numpy scipy pandas pyyaml imageio pillow opencv-python \
    einops joblib matplotlib plyfile scikit-image tqdm tensorboard
```

> **Gotcha:** in a non-interactive shell `conda activate wrfgsplus` can silently no-op (you stay on the base `/usr/bin/python`, which lacks `plyfile`/the rasterizer, so `scene/__init__.py` fails to import). Invoke the interpreter by **full path** (`~/miniconda3/envs/wrfgsplus/bin/python`) instead of relying on `activate`.

## 2. Build the three CUDA extensions

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=10.0          # B200 = sm_100
$PY -m pip install --no-build-isolation --no-cache-dir \
    ./submodules/simple-knn \
    ./submodules/fused-ssim \
    ./submodules/diff-gaussian-rasterization

# sanity-check the built arch (should print sm_100 for all three)
SP=~/miniconda3/envs/wrfgsplus/lib/python3.11/site-packages
for m in diff_gaussian_rasterization simple_knn; do cuobjdump $SP/$m/_C*.so | grep -m1 'arch = sm_'; done
```

> **Different GPU?** `submodules/diff-gaussian-rasterization/setup.py` hard-codes
> `-gencode arch=compute_100,code=sm_100`. On a non-Blackwell GPU (e.g. A100 = sm_80)
> change that line to your arch (`code=sm_80`) or the build will produce code your GPU
> can't run. simple-knn / fused-ssim pick up `TORCH_CUDA_ARCH_LIST` and need no edit.

## 3. (Optional) regenerate the dataset

The converted dataset ships in the experiment archive; regenerate only if you have the
raw Multimodal-Wireless drop (`/NHNHOME/.../multimodal_wireless`). The exact params used
for the final experiment:

```bash
PYTHONPATH=tools $PY -m mw2wrfgs.cli --out-dir ./data_mw_town05_parkinglot \
    --town Town05 --scenario Town05_parkinglot \
    --sigma-az 30 --sigma-el 30 --floor-abs 0.50 --seed 0
# tx_pos scale is auto-measured = 75.486 m (max CAV–RSU distance)
```

> Training uses an **unseeded** Gaussian init (`gaussian_init()`), so a trained model is
> **not** bit-reproducible from code+data — keep the checkpoint to recover an exact model.

## 4. Train / resume / evaluate

```bash
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# train from scratch
$PY train.py --datadir ./data_mw_town05_parkinglot --disable_viewer

# resume / re-evaluate from a saved checkpoint
$PY train.py --datadir ./data_mw_town05_parkinglot \
    --start_checkpoint <path>/chkpnt200000.pth --disable_viewer
```

Outputs: checkpoints/point clouds under `output/<timestamp>/`, metrics + comparison PNGs
under `logs/<timestamp>/` (both git-ignored).

**Final experiment (reference):** Town05_parkinglot, 200k iters → median SSIM **0.9992**
(660 test) vs mean-spectrum baseline **0.9624**; final 15,288 Gaussians. The checkpoint,
dataset, metrics and a file-by-file README are bundled in a local `final_experiment_archive/`
(git-ignored — kept off the repo).

## 5. Source fixes that make the B200 build work

All committed in *"Port WRF-GS+ build to B200 (sm_100) / CUDA 13 / torch 2.x"*:

- `submodules/diff-gaussian-rasterization/setup.py` — force `-gencode arch=compute_100,code=sm_100`. **This is the actual fix for the OOM**: without an sm_100 SASS target the B200 JIT-compiled sm_75 PTX from nvcc 13.1 and failed with CUDA error 222 ("PTX compiled with an unsupported toolchain"); every device/cub kernel then returned garbage (negative `num_rendered`) and the rasterizer tried to allocate ~10¹⁴ GiB.
- `cuda_rasterizer/auxiliary.h` — replace glibc `_Float32` macros `M_1_PIf32`/`M_2_PIf32` with identical plain-`float` constants (nvcc 13 ICEs on the C++23 extended-float type).
- `cuda_rasterizer/rasterizer_impl.h` — add `<cstdint>` (GCC 13 / CUDA 13 no longer pull it in transitively).
- `cuda_rasterizer/forward.cu` — defensive cull of non-finite / non-positive-definite 2D covariances (equirectangular pole/near-camera singularities) so a degenerate splat can't overflow the tile rectangle.
- `train.py` — log `dataset.datadir` (the removed dead `datadir` variable was still referenced by a logger line → `NameError`).

## Related docs
- Design spec: [docs/superpowers/specs/2026-06-17-wrfgs-multimodal-wireless-adaptation-design.md](superpowers/specs/2026-06-17-wrfgs-multimodal-wireless-adaptation-design.md)
- Implementation plan: [docs/superpowers/plans/2026-06-17-wrfgs-multimodal-adaptation.md](superpowers/plans/2026-06-17-wrfgs-multimodal-adaptation.md)
