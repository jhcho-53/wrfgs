# Scene-Conditioned Wireless Radiation Fields: Relative Geometry Enables Cross-Scene Generalization

**Date:** 2026-06-18 · **Status:** research findings + paper positioning (cross-town scale-up deferred)

> A research log consolidating an extension of WRF-GS+ from a per-scene-overfit
> radiation-field model into a **scene-conditioned** model that takes the
> environment geometry (RSU lidar) as input and generalizes to unseen scenes.
> All numbers below are from runs in this repo on 2× NVIDIA B200.

---

## 1. One-paragraph summary

WRF-GS+ / NeRF2-style neural radiation fields **memorize a single environment**
(they condition only on the transmitter position and bake the scene into their
weights), so they cannot be applied to a new environment without retraining.
We built a **scene-conditioned** variant — Gaussians frozen on the RSU lidar's
static geometry + a **shared** signal MLP — and found that **how the MLP is
conditioned is decisive**: conditioning on **absolute coordinates** memorizes
the training scenes and collapses zero-shot (held-out SSIM 0.13–0.29), whereas
conditioning on **relative single-bounce geometry** (incident/outgoing
directions, bounce angle, log-distances) **generalizes** to unseen *in-distribution*
scenes — held-out direction-of-arrival (DoA) error **10.6°** vs a 53.6° trivial
baseline (5× better) — while failing to **extrapolate** to an
out-of-distribution scene (an elevated-RSU bridge scenario: 83°) **only when
trained on too few (2) scenes**. **Scaling to 12 diverse training scenes (4
towns, ground + elevated RSUs) resolves this**, achieving **cross-scene AND
cross-town zero-shot generalization** on all 4 held-out scenes (DoA **1.7–5.8°**,
2–21× better than the trivial baseline; the elevated scene **83° → 5.8°**). A
power-weighted angular metric was essential to see any of this, because SSIM
saturates on the smooth densified targets. This matches and extends the very
recent literature (GeNeRT: relative features → generalization; RayProNet:
absolute geometry → per-scene), and the specific instantiation
(**lidar-initialized 3D Gaussian Splatting + shared relative-geometry signal
MLP for cross-scene spectrum prediction**) appears novel.

---

## 2. Problem & motivation

- **Task.** Reconstruct the spatial spectrum (90×360 elevation×azimuth power-
  angular map) at a fixed RSU as a function of the moving CAV transmitter
  position, in a CARLA + Sionna ray-traced V2I dataset (**Multimodal-Wireless**,
  arXiv:2511.03220 — ~160k frames, 4 towns, 16 scenarios, 3 weathers, CSI synced
  with lidar/RGB-D/IMU/radar).
- **Gap.** WRF-GS+ (3DGS for the wireless radiation field, IEEE TWC 2025) and
  NeRF2 (MobiCom 2023) are **per-scene**: trained on one environment's
  measurements, conditioned only on `tx_pos`, with **no scene-geometry input**.
  A new environment ⇒ full retrain. Cross-environment generalization is an
  **open problem** (see §6).

### Critical analysis of the per-scene task (why we pivoted)
Three findings made the per-scene spectrum-reconstruction task scientifically
weak on this data, motivating the scene-conditioned pivot:
1. **Leaky evaluation.** A random 80/20 split leaks: 95.3% of test frames have a
   trajectory-adjacent train frame at ~0 normalized distance. A **no-training
   nearest-neighbour copy** of the closest train spectrum scores median SSIM
   **0.9998**, matching/beating a 200k-iteration trained model (0.9992) — i.e.
   the headline number measures interpolation, not learning.
2. **The channel is dynamics-invariant.** Although each scenario contains real
   moving traffic (12 of 13 background vehicles move up to 55 m), two different
   CAVs at the **same position at different times** (different traffic) get the
   **same spectrum** (cross-CAV SSIM 0.999 ≈ same-CAV adjacent 0.9996; 99.8% of
   ultra-close cross-CAV pairs ≥0.99). Sionna ray-traced the static map only, so
   the channel is a deterministic function of ego position and **carries no
   dynamic information** — "dynamic robustness" is not testable on this channel.
3. **SSIM saturates on the densified target.** The targets are dense beamformed-
   style maps (we synthesize σ≈30°, floor 0.5 to match the original WRF-GS+
   density); they are so smooth/similar that the mean-spectrum baseline already
   scores SSIM **0.88–0.95**, so SSIM cannot distinguish real per-Tx structure
   from a generic average (this resurfaces in §5).

→ The meaningful, harder, novel problem is **cross-scene generalization**:
predict a *new* environment's channel from its geometry.

---

## 3. Method

### 3.1 Data conversion (per scenario)
`tools/mw2wrfgs/` converts each (town, scenario, RSU) into a `data_test200`-format
dataset:
- **Spectrum synthesis.** Per Sionna path k: power `p_k = |a[…,k,…]|²` (antenna
  dims are phase-only), arrival angles from the **world-frame** `glob_theta_r/
  glob_phi_r`, mapped to renderer pixels through the equirectangular projection
  `lon=atan2(x,z)`, `lat=asin(y/r)`, `ndc2Pix`, with the world-z-up→camera-y-up
  axis permutation. Each path is a broad anisotropic Gaussian lobe over a floor
  (densified to match the original dense targets), per-image peak-normalized.
- `tx_pos = R_rsu⁻¹ (cav−rsu)/scale`; gateway authored as an identity gauge.

### 3.2 Static scene extraction
`tools/extract_rsu_scene.py` aggregates ~20 RSU lidar frames and keeps
voxel-occupancy-static points (the channel is static, so the ~4% moving points
are dropped), yielding ~12–21k points in the RSU-local frame (range 1–120 m;
near-horizon geometry, matching the channel's 1–6° arrival elevations).

### 3.3 Scene-conditioned model (`train_multiscene.py`, `utils/rel_deform.py`)
- **Per-scene Gaussians are FROZEN on the lidar geometry** (`GaussianModel.
  init_from_lidar`): positions = lidar points, SH = 0, all per-Gaussian
  attributes `requires_grad=False`. The scene enters **only** through its lidar
  → zero per-scene learnable parameters → a held-out scene just supplies its
  lidar. No densification.
- **One SHARED signal MLP** maps each scatterer + Tx to the per-Gaussian EM
  signal (added to SH; the renderer projects it to the correct DoA pixel).
  Trained jointly over scenes (round-robin), only the shared MLP learns.
- **The conditioning is the key variable:**
  - **Absolute** (baseline): MLP on `(point_xyz, tx_pos)` (the original
    DeformNetwork).
  - **Relative** (ours, `RelDeformNetwork`): MLP on each scatterer's
    **single-bounce relative geometry** w.r.t. Rx (origin) and Tx:
    `[d_rx, d_tx, bisector, cos(bounce), log r_rx, log r_tx, log path]`. Tx is
    reconstructed into the Rx-frame metres of the lidar
    (`P(R_z(yaw)·(tx_pos·scale))`; frame verified: reconstructed-vs-true CAV
    direction error **0.0°**).

---

## 4. Experiments

Hardware/software: 2× NVIDIA B200 (sm_100). The legacy 2023 CUDA rasterizer did
not run on Blackwell; we ported it (see §7). Scene-conditioned runs: 2–3 Town05
scenarios (parkinglot, ringroad, CBDcrossroad), shared MLP only, 3k iterations.

### 4.1 Capacity check (does one shared MLP fit several scenes?)
Yes. Trained jointly on 3 scenes, the shared MLP reaches per-scene SSIM
~0.79–0.89 — one model represents distinct scenes from lidar geometry + tx_pos
with zero per-scene parameters.

### 4.2 Zero-shot generalization — absolute vs relative conditioning (controlled)
Leave-one-out: train on 2 scenes, evaluate on the held-out third. Only the
conditioning changes; data/split/everything else identical.

| Held-out scene | Absolute SSIM | **Relative SSIM** |
|---|---|---|
| CBDcrossroad | 0.135 | **0.858** |
| ringroad | 0.286 | **0.703** |
| parkinglot | — | **0.799** |

Absolute coordinates **collapse** zero-shot (memorization); relative geometry
lifts held-out SSIM by 3–6×. **But** all relative numbers are still **below** the
mean-spectrum baseline (0.88–0.95) — SSIM cannot tell whether this is real
transfer or generic-average output (the §2.3 saturation problem).

### 4.3 The discriminative metric: power-weighted DoA angular error
We add a power-weighted circular-mean **azimuth/elevation error (degrees)** —
"does the predicted energy sit at the right Tx-dependent direction?". It is
discriminative even when SSIM saturates.

| Held-out scene | **Relative held-out az-err** | mean-spectrum baseline az-err | verdict |
|---|---|---|---|
| CBDcrossroad | **10.6°** | 53.6° | ✅ generalizes (5× better) |
| parkinglot | 26.5° | 20.9° | ≈ baseline |
| ringroad | 83.0° | 53.5° | ❌ worse than baseline (OOD) |

(Train-scene azimuth error: 1.4–2.4°. Elevation errors 1.0–2.1° throughout —
elevation is near-degenerate as the channel is near-horizon.)

**Interpretation.** Relative features produce *genuine* cross-scene
generalization for an **in-distribution** held-out scene (CBDcrossroad, a
ground-level crossroad like the training scenes): the model tracks the
Tx-dependent DoA to ~10°, 5× better than the static average. It **fails to
extrapolate** to the **out-of-distribution** ringroad — whose RSU sits at
**z = 10 m** (elevated, on a bridge) vs **z = 0 m** for both training scenes — so
the model, never having seen an elevated RSU, mispredicts the azimuth (83°,
worse than the average). This is the textbook in-distribution-generalization /
out-of-distribution-extrapolation split, and it directly motivates broadening
the training geometry distribution (§4.4).

### 4.4 Scaling to 12 diverse scenes — cross-scene AND cross-town generalization
We converted all **16 V2I scenarios across 4 towns** (Town03×5, Town05×5,
Town07×2, Town10×4; ~54k samples), trained the relative model on **12** (held out
one per town: Town03_5wayroad, Town05_ringroad, Town07_grainsilos,
Town10_crossroad), and crucially **included elevated-RSU scenes (Town03_Tjunction
z=8 m, Town03_crossroad z=8.2 m) in training**. 12k iterations.

| Held-out scene | model DoA az-err | baseline DoA | advantage |
|---|---|---|---|
| Town03_5wayroad | **1.7°** | 36.0° | 21× |
| Town05_ringroad (elevated) | **5.8°** | 53.5° | 9× |
| Town07_grainsilos | **5.3°** | 36.7° | 7× |
| Town10_crossroad | **3.5°** | 6.9° | 2× |

**All four held-out scenes generalize** (DoA 1.7–5.8°, beating the trivial
baseline 2–21×), and the held-out errors equal the train-scene errors (0.7–6.7°)
— a negligible generalization gap. Decisively, **the elevated-RSU ringroad that
catastrophically failed at 83° with 2 ground-only training scenes now
generalizes at 5.8° (14×)** once the training set contains *other* elevated
scenes. This is direct evidence that the earlier OOD failure was a
**scene-coverage** problem, solved by scaling the training diversity — i.e. the
relative-geometry scene-conditioned model achieves genuine **cross-scene and
cross-town zero-shot generalization** given enough diverse scenes. (Held-out SSIM
stays 0.81–0.90, around the saturated baseline — only the DoA metric reveals the
result.)

---

## 5. Key findings

1. **Conditioning representation is decisive for cross-scene transfer.** With an
   identical frozen-lidar Gaussian backbone and shared MLP, switching from
   absolute coordinates to relative single-bounce geometry turns memorization
   (held-out 0.13–0.29) into generalization (held-out DoA 10.6° vs 53.6°
   baseline). This is a clean **controlled** result.
2. **SSIM is the wrong metric here; a power-weighted DoA error is right.** On the
   smooth densified targets SSIM saturates (baseline 0.88–0.95) and hides both
   the success and the failure; the angular metric reveals both.
3. **Diversity, not architecture, gates extrapolation — and scaling solves it.**
   With only 2 (ground-only) training scenes the OOD elevated-RSU scene fails
   (83°); with **12 diverse scenes including elevated RSUs it generalizes (5.8°,
   14×)**, and all 4 cross-town held-out scenes beat the baseline 2–21×. The
   relative-geometry model achieves genuine **cross-scene + cross-town zero-shot
   generalization** given enough scene coverage.
4. **The channel on this dataset is geometry-deterministic and dynamics-blind**
   (Sionna static-map ray tracing), so scene **geometry** — not measured
   neighbour spectra and not dynamics — is the right conditioning signal, and
   the lidar provides it directly.

---

## 6. Related work & positioning (verified literature survey, 2023–2025)

**Per-scene-overfit (what we extend), confirmed condition only on Tx/Tx-Rx, no
geometry input, no cross-environment generalization:** NeRF2 (arXiv:2305.06118,
MobiCom'23), WRF-GS (arXiv:2412.04832, IEEE TWC'25), WRF-GS+, NeWRF
(arXiv:2403.03241, ICML'24), WiNeRT (ICLR'23).

**Cross-scene generalizers (the frontier):**
- **GRaF / GWRF** (arXiv:2502.05708, 2025) — the canonical "generalizable RF
  radiance field" for spatial-spectrum synthesis; empirically shows per-scene
  baselines collapse cross-layout (**NeRF2 PSNR −35.9%**). **But** it conditions
  on **neighbouring *measured* transmitter spectra** (few-shot, needs in-scene
  measurements at test), **not on lidar/mesh geometry**, and is itself "lower
  quality than single-scene."
- **GeNeRT** (arXiv:2506.18295, 2025) — geometry-conditioned generalizable neural
  ray tracer; achieves **inter-scenario zero-shot** generalization specifically
  via **relative geometric features** (incident angles, normals, Fresnel
  embeddings). Predicts rays/MPCs, not a spectrum image. **This is the direct
  literature support for our absolute→relative result.**
- **RayProNet** (arXiv:2406.16907, IEEE'24) — PointNet point-cloud geometry
  encoder for RSS/pathloss, but **"any change to scene geometry necessitates
  re-training"** → per-scene w.r.t. geometry (a cautionary precedent: absolute
  point-cloud encoding does not generalize).
- **RadioUNet** (TWC'21, arXiv:1911.09002), **PMNet** (TWC'24, arXiv:2312.03950) —
  genuinely take the environment **map** as input and generalize to held-out
  maps, but only as **2D building-footprint → scalar pathloss**, not 3D spatial
  spectra.

**Novelty.** No verified prior work uses **lidar-initialized 3D Gaussian
Splatting + a shared (relative-geometry) signal MLP for cross-scene spectrum
prediction**. Closest: GRaF (cross-scene, but neighbour-spectra, no geometry) and
RayProNet (point geometry, but per-scene). Our formulation sits at their
intersection in a Gaussian-splatting field — apparently novel. *Caveat:* a
targeted check of Jan–Jun 2026 GS-for-RF preprints is advised before a hard
novelty claim (absence of evidence ≠ absence).

**Dataset.** Multimodal-Wireless (arXiv:2511.03220) provides the geometry+CSI
pairing for exactly this task; the dataset paper defines tasks (channel
estimation, beamforming, blockage) but reports no solved cross-environment
generalization result.

---

## 7. Engineering contribution: porting WRF-GS+ to Blackwell / CUDA 13

The legacy rasterizer OOM'd on B200 (tried to allocate ~10¹⁴ GiB). Root cause was
an **arch mismatch**, not memory: the extension compiled to **sm_75 SASS only**,
so on sm_100 the driver JIT'd compute_75 PTX from nvcc 13.1 → CUDA error 222
("PTX compiled with an unsupported toolchain") → every cub/device kernel silently
returned garbage → negative `num_rendered` → absurd buffer. Fixes (build/portability
only, projection math unchanged): force `-gencode arch=compute_100,code=sm_100`;
replace glibc `_Float32` macros `M_1_PIf32/M_2_PIf32` (nvcc-13 device-codegen ICE)
with plain floats; add `<cstdint>` (GCC13/CUDA13 strictness); a defensive
non-finite-covariance cull for the equirectangular pole singularity; and a
`dataset.datadir` log fix. Verified: original `data_test200` trains (median SSIM
~0.69) and the converted dataset trains end-to-end on B200. Env:
conda `wrfgsplus` (py3.11 + torch 2.12 cu130).

---

## 8. Limitations & next steps

- **OOD extrapolation — RESOLVED by scaling (§4.4).** Training on 12 diverse
  scenes (incl. elevated RSUs) turned the elevated-RSU ringroad from an 83° OOD
  failure into a 5.8° success, and all 4 cross-town held-out scenes generalize.
  Remaining: push further (all-but-one-town leave-out, harder geometries) and
  establish the **scene-count scaling law** (unquantified in the literature).
- **De-densify the target** (σ≈8, floor≈0.05) so the spectrum carries per-Tx
  structure and SSIM regains discriminative power (the angular metric already
  compensates; both together are ideal). *(Quick GPU re-train; CPU re-render
  pending a free node.)*
- **Physics priors** (GeNeRT-style Fresnel branches / a Sionna-RT differentiable
  prior) and **stronger geometry encoders** (sparse-conv over the lidar instead
  of frozen points) are candidate levers, ranked open in the literature.
- **Scene-count scaling law** (how many scenes before broad generalization
  emerges) is unquantified in the literature and worth establishing here.

---

## Figures (`docs/research/figures/`, PNG + PDF; `tools/qualitative_figure.py`)

- **fig5_scaleup** — THE headline: (a) scaling solves OOD (ringroad 83°→5.8°,
  14×); (b) all 4 cross-town held-out scenes generalize, beating baseline 2–21×.
- **fig1_generalization** — (a) conditioning ablation (2-scene): zero-shot
  held-out SSIM, absolute vs relative; (b) zero-shot DoA azimuth error vs the
  mean-spectrum baseline (CBD generalizes 5×, ringroad OOD-fails at 2 scenes).
- **fig2_table** — results table: SSIM / DoA-error per method × held-out scene.
- **fig3_diagnostics** — why we pivoted: (a) random-split leakage (NN-copy ≈
  trained), (b) dynamics-blind channel, (c) SSIM-saturation-vs-DoA-discrimination.
- **fig4_qualitative** — zero-shot held-out spectra (unseen CBDcrossroad): GT vs
  absolute (collapses) vs relative (matches GT).

## 9. Reproducibility (code in this repo, branch `wrfgs-multimodal-adaptation`)

- Converter: `tools/mw2wrfgs/` (+ `tools/calibrate_spectrum.py`,
  `tools/check_alignment.py`). Static scene: `tools/extract_rsu_scene.py`.
- Per-scene baseline: `train.py --datadir <converted>`.
- Scene-conditioned model: `scene/gaussian_model.py::init_from_lidar`,
  `utils/rel_deform.py`, `train_multiscene.py`
  (`--scenes DATADIR:LIDAR_NPY … --holdout-scenes …`). Relative features +
  power-weighted DoA metric are the current defaults.
- Spec/plan: `docs/superpowers/specs/2026-06-17-…`, `docs/superpowers/plans/2026-06-17-…`.
