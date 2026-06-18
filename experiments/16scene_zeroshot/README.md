# 16-scene scale-up — cross-scene + cross-town zero-shot generalization

Durable record of the experiment behind commit `371d91b` and §4.4 of
[docs/research/2026-06-18-scene-conditioned-wrf-generalization.md](../../docs/research/2026-06-18-scene-conditioned-wrf-generalization.md).
Created to preserve what previously lived only in volatile `/tmp` and in prose.

## What it is

The relative-geometry **scene-conditioned** WRF-GS+ model (Gaussians frozen on each
scene's static RSU-lidar geometry + a single **shared** `RelDeformModel`) trained
jointly on **12** V2I scenarios and evaluated **zero-shot** on **4** held-out
scenes (one per town, never trained). "16 scenes" = 12 train + 4 holdout = the 16
converted datasets total.

- Model/code: [train_multiscene.py](../../train_multiscene.py),
  [utils/rel_deform.py](../../utils/rel_deform.py),
  `GaussianModel.init_from_lidar` ([scene/gaussian_model.py](../../scene/gaussian_model.py)).
- Metric: **power-weighted DoA azimuth error (deg)** is the headline. SSIM saturates
  (~0.88 baseline) on these smooth densified targets and is non-discriminative.

## Results (original run, iteration 12000)

Zero-shot held-out (machine-readable in [results.json](results.json); raw log in
[archive/train16.log](archive/train16.log)):

| Held-out scene | model DoA az | baseline DoA | advantage |
|---|---|---|---|
| Town03_5wayroad | **1.7°** | 36.0° | 21× |
| Town05_ringroad (elevated RSU) | **5.8°** | 53.5° | 9× (vs 2-scene 83° failure → 14×) |
| Town07_grainsilos | **5.3°** | 36.7° | 7× |
| Town10_crossroad | **3.5°** | 6.9° | 2× |

Held-out errors ≈ train-scene errors (0.7–6.7°): negligible generalization gap.
The elevated-RSU ringroad that failed at 83° with only 2 ground-only training
scenes generalizes at 5.8° once elevated scenes are in the 12-scene training set.

## Reproduced run (committed checkpoint, `seed=0`)

Re-trained here from scratch with `run_16scene.sh` (seed 0, 12000 iters) and a
**durable saved checkpoint**: `runs/retrain_seed0_20260618_130123/deform_latest.pth`
(2.2 MB), with `results.json` + `train.log` alongside. The finding reproduces (DoA
matches/beats the original):

| Held-out scene | reproduced DoA az (seed 0) | original DoA az |
|---|---|---|
| Town03_5wayroad | 1.1° | 1.7° |
| Town05_ringroad (elevated) | 3.7° | 5.8° |
| Town07_grainsilos | 2.7° | 5.3° |
| Town10_crossroad | 1.5° | 3.5° |

`--eval-only --load-checkpoint deform_latest.pth` reloads this checkpoint and
reproduces these exact numbers without retraining (verified bit-exact: eval is
`no_grad` + `shuffle=False`).

## Files

- `results.json` — authoritative machine-readable results of the original run (no seed).
- `run_16scene.sh` — reproducible launcher: `bash experiments/16scene_zeroshot/run_16scene.sh [GPU] [ITERS]`
  (defaults GPU 0, 12000 iters, `SEED=0`). Writes a checkpoint + `results.json` +
  `train.log` under `experiments/16scene_zeroshot/runs/retrain_seed<seed>_<ts>/`.
- `archive/` — preserved raw stdout logs that previously existed only in `/tmp` (tmpfs,
  lost on reboot): `train16.log` (the 16-scene run), `train16.scenes.txt` (the
  `--scenes` arg string), plus the supporting 2-/3-scene controlled runs (`ms_fold_*`,
  `rel_hold*`, `ang_*`) backing doc §4.2/§4.3.
- `runs/` — outputs of re-training (checkpoints `deform_*.pth`, `results.json`, `train.log`).

## Reproduce

```bash
# prereqs (both .gitignored, regenerate if absent):
#   data_mw_<Town>_<scenario>/  via tools/mw2wrfgs
#   scenes_static/<scene>.npy   via tools/extract_rsu_scene.py
bash experiments/16scene_zeroshot/run_16scene.sh 0 12000      # train 12, eval 4 zero-shot

# reload the trained model and re-evaluate WITHOUT retraining:
python train_multiscene.py --eval-only \
  --load-checkpoint experiments/16scene_zeroshot/runs/<run>/deform_latest.pth \
  --out-dir experiments/16scene_zeroshot/runs/<run>/eval \
  --scenes <12 DATADIR:LIDAR_NPY ...> --holdout-scenes <4 DATADIR:LIDAR_NPY ...>
```

**Reproducibility caveats.** The original run set **no seed**, so re-runs match
statistically, not bit-exactly. `train_multiscene.py` now sets `--seed` (default 0),
saves a `deform_latest.pth` checkpoint, and dumps `results.json`; `--iterations 12000`
must be passed explicitly (script default is 200000). The `data_mw_*` and
`scenes_static/` inputs are not in git (gitignored) — regenerate from the raw
Multimodal-Wireless drop with the `tools/` pipeline.
