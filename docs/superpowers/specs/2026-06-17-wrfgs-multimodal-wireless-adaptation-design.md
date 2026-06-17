# WRF-GS+ → Multimodal-Wireless (V2I) Adaptation — Design

- **Date:** 2026-06-17
- **Status:** Approved design (pre-implementation)
- **Author:** Claude Code session, with the user
- **Related code:** this repo (WRF-GS+ fork of 3DGS); dataset at `/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless`
- **Dataset paper:** *Multimodal-Wireless* (CARLA + Sionna), arXiv:2511.03220

> Every load-bearing technical claim below was independently produced and adversarially re-verified by a multi-agent workflow against the CUDA rasterizer, the Python pipeline, and the actual dataset files. File:line citations are from this repo unless noted.

---

## 1. Goal & scope

Apply the existing WRF-GS+ model — **unchanged in architecture** — to the Multimodal-Wireless dataset so it reconstructs the **spatial spectrum at a fixed roadside unit (RSU)** as a function of a moving vehicle's (CAV's) transmitter position.

**In scope (MVP):**
- A single **(weather, town, scenario, antenna-config, RSU)** unit → one per-scene WRF-GS+ model, exactly mirroring the original per-gateway training.
- Fixed first target: **V2I, weather=`sunny`, antenna=`Nt_1_64_Nr_1_16_fc_28GHz`, Town05 / `Town05_parkinglot` / `rsu_1`** (any single V2I scenario works; this is the worked example).
- Prediction target: a **densified 2D Direction-of-Arrival (DoA) spatial-spectrum image**, 90×360, synthesized from Sionna ray-tracing paths.

**Out of scope (future):** V2V (moving RX breaks the fixed-gateway gauge), cross-scene / cross-weather generalization, multi-RSU conditioning, other antenna configs, beam-prediction (the dataset paper's own task).

---

## 2. How the WRF-GS+ pipeline actually works (verified)

This is a 3DGS fork repurposed for radio. The mechanics that make the adaptation valid:

1. **The rasterizer is a modified equirectangular (lon–lat) fork, not a pinhole.** In `preprocessCUDA` the only projection is `point3ToLonlatScreen` (`submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:465`), computing `lon = atan2f(x, z)` and `lat = asinf(y / r)` (`auxiliary.h:230-231`). The perspective `projmatrix` path is explicitly commented `// useless` (`auxiliary.h:168`); culling is by `too_close()` with `rr <= 0.04` using only the view matrix (`forward.cu:460`). `backward.cu:176` notes the perspective limitation "should not be" present "in panoramic". **Therefore `FoVx=FoVy=180°` (`tan(90°)=∞`) causes no degeneracy — it only feeds the unused projmatrix path — and a 90×360 angular image is a structurally valid render target.**

2. **Exact pixel↔angle map of the renderer:**
   - `col = ndc2Pix(lon/π, W) = ((lon/π + 1)·W − 1)/2`, with `lon = atan2(x_cam, z_cam) ∈ [−π, π]`. So `lon=−π → col≈0`, `lon=0 → col≈180`, `lon=+π → col≈360` (azimuth wraps at the col-0/col-360 seam).
   - `row = ndc2Pix(lat·2/π, H) = ((lat·2/π + 1)·H − 1)/2`, with `lat = asin(y_cam/r) ∈ [−π/2, +π/2]`. So **`lat=0` (horizon) → row≈45 (image centre)**, up-pole → row≈89, down-pole → row≈0. The row axis is **signed elevation**, not zenith.
   - The `computeCov2D` Jacobian (`forward.cu:84-99`, using `W·0.5/π` and `H/π`) is the analytic Jacobian of this same lon/lat map.

3. **Output → loss path.** The model renders a 2-channel image; `pred = |real + i·imag|` (`train.py:160-163`); loss = `(1−λ)·L1 + λ·(1−SSIM)` vs the single-channel real GT magnitude (`train.py:168-174`, `λ=lambda_dssim=0.2`). Eval is identical (`train.py:267-278`). **GT is and stays a real magnitude image in [0,1]; no complex GT is needed.**

4. **The camera is a single fixed pose; `tx_pos` never moves it.** `tx_pos` flows only into the DeformNetwork as the D-NeRF-style conditioning input `time_input` (`train.py:136-140`); the camera is rebuilt every iteration from the constant `scene.r_o` (position) and `scene.gateway_orientation` (quaternion) via `generate_new_cam(R, r_o)` (`utils/generate_camera.py:8-20`, `scene/cameras.py:53`). All per-frame variation is encoded by the deformation field conditioned on `tx_pos`.

5. **DeformNetwork conditioning scale matters.** `tx_pos` is positionally encoded with `t_multires=6` → frequency bands `[1,2,4,8,16,32]` (`utils/pos_utils.py:65,41-44`; `is_blender=True` via `scene/deform_model.py:11`). Inputs much beyond ~unit magnitude alias the highest band `sin(32·x)`. The reference `tx_pos` (RFID) has abs-max ≈1.07, norms 0.86–1.33 (`data_test200/tx_pos.csv`).

---

## 3. Dataset structure & conceptual mapping

**Channel data** (`<root>/<weather>/Channel Data/V2I/Nt_1_64_Nr_1_16_fc_28GHz/<Town>/<scenario>/cav_k/NNNNNN_paths.npz`, where `<weather>` ∈ {sunny, rainy, foggy}; note the channel path has **no `_seed` suffix** on `<scenario>`) are Sionna `Paths` exports. Keys & shapes (verified):
- `a` — complex64 `(1, 1, Nr=16, 1, Nt=64, n_paths, 1)` = `[batch, n_rx, n_rx_ant, n_tx, n_tx_ant, paths, time]`. **`|a|` is constant across all 16×64 antenna elements** (max ratio 1.0000005); the antenna axes carry only steering **phase**, not magnitude.
- `tau` — path delays `(1,1,1,n_paths)`.
- `theta_r`, `phi_r` — RX arrival **zenith** ∈[0,π] and **azimuth** ∈[−π,π], in the RX-**local** frame.
- `theta_t`, `phi_t` — TX departure angles.
- `glob_theta_r/phi_r`, `glob_theta_t/phi_t` — the same angles in the **world** frame.
- `n_paths` is **variable** — read it per file from `a.shape[5]`; never assume a fixed count. Dataset-wide it is ≈1–9 (mean ≈3.5); the worked scenario Town05_parkinglot is **sparser** (≈1–5, mean ≈2.4), so size buffers from the actual array, not from 9. Padded/zero-power slots (`|a|==0`) occur and must be masked out before use.

**Sensor/pose data** live under a sibling `Sensor Data` tree whose `<scenario>` carries a **`_seed*` suffix** (e.g. `Town05_parkinglot_seed42`) — an asymmetry with the no-seed Channel Data path that the converter must reconcile (§8). Per-frame pose: `<root>/<weather>/Sensor Data/<Town>/<scenario>_seed*/cav_k/NNNNNN.yaml` (6-digit frame number) → ground-truth CAV transmitter position at **`sensors.vehicle_pose.location {x,y,z}`** (top-level YAML keys are `[actor, frame, sensors, vehicles]`; use `sensors.vehicle_pose` — NOT `sensors.GPS` / `sensors.predicted_ego_pos`). **RSU pose** is in `<root>/<weather>/Sensor Data/<Town>/<scenario>_seed*/config.yaml` → `scenarios.<scenario>.rsu_transform { location, rotation.yaw }` (e.g. Town05_parkinglot: location `(-59, 11, 0)`, yaw 45°).

**Frame join is exact:** channel and sensor each have the same 1100 frames per cav (e.g. indices 280–1379), 100% intersection. Join on `(town, scenario, cav, frame)`.

| WRF-GS+ (original RFID) | Multimodal-Wireless (V2I) |
|---|---|
| Fixed gateway RX antenna | **RSU** (`rsu_transform`, fixed per scenario) |
| Moving transmitter `tx_pos` | **CAV vehicle position** (`sensors.vehicle_pose.location`) |
| Gateway orientation quaternion | RSU pose (we use an **identity gauge** — see §6) |
| 90×360 spatial-spectrum PNG | **Synthesized densified 2D DoA image** (§5) |

---

## 4. Sample definition

- **One sample = one `(cav, frame)` pair.** The converter **enumerates the `cav_*` directories** present for the scenario (do not hardcode `cav_1..cav_3`); for the worked scenario this is 3 cavs × 1100 frames = **3,300 samples** (verified to exist). Exact count = intersection of available channel+pose frames; recompute per scenario.
- Conditioning input `tx_pos` = that cav's `sensors.vehicle_pose.location` at that frame (transformed per §6).
- Target = the densified 2D DoA spectrum at the RSU for that cav/frame (§5).
- **Train/test split — owned by the converter (single owner, see §7).** The converter writes a random **80/20** `train_index.txt` / `test_index.txt` over the generated sample IDs. Because these files then exist, `Scene`'s built-in auto-split (`scene/__init__.py:63-64`, which only runs when the index files are absent) is intentionally skipped. (Pitfall: if you ever rely on the built-in `split_dataset` instead, you MUST pass `ratio=0.8` — its default is `0.1` and it assigns the **first** `ratio·N` IDs to train, so the default would put 10% in train.)

---

## 5. Spectrum synthesis — densified 2D DoA (the core of this work)

For each sample, build a `90×360` float image `S`, then save as 8-bit PNG.

### 5.1 Per-path power
For each valid path `k` (skip `|a|==0` slots), use the intrinsic per-path power
```
p_k = |a[0, 0, 0, 0, 0, k, 0]|^2
```
The antenna dims are phase-only, so any rx/tx reduction equals `p_k` up to a constant that cancels under normalization. Document this; do **not** describe it as "array gain".

### 5.2 Angles → direction → pixel (uses GLOBAL angles + explicit axis remap)
Use the **world-frame** angles `glob_theta_r`, `glob_phi_r` (each shape `(1,1,1,n_paths)`; index as `glob_theta_r[0,0,0,k]`). Do **not** use the RX-local `theta_r/phi_r`: the local azimuth frame is re-oriented per frame (a per-frame constant rotation offset that varies by tens of degrees across frames, e.g. −92°/±180°), so local angles are not consistent across samples; only the world frame is.

Build the world unit direction of arrival (Sionna physics convention, z-up):
```
θ = glob_theta_r[0,0,0,k]   # zenith ∈ [0,π]
φ = glob_phi_r[0,0,0,k]     # azimuth ∈ [−π,π]
d_world = [ sinθ·cosφ, sinθ·sinφ, cosθ ]
```
Map into the renderer's camera-axis convention via a **fixed axis permutation** `P` chosen so that world-up (z) becomes the renderer's elevation axis (camera +y):
```
d_cam = [ d_world.x, d_world.z, d_world.y ]      # camera_x=world_x, camera_y=world_z, camera_z=world_y
```
Apply the renderer's own formulas (mirrors §2.2 exactly):
```
lon = atan2(d_cam.x, d_cam.z)                    # ∈ [−π, π]
lat = asin(d_cam.y / |d_cam|)                    # ∈ [−π/2, +π/2]  (here |d_cam|=1)
col_c = ((lon/π + 1)·360 − 1)/2                  # fractional centre column
row_c = ((lat·2/π + 1)·90  − 1)/2                # fractional centre row
```
This **fixes the draft's row flip** (elevation is horizon-centred at row≈45, not zenith-from-row-0) and the azimuth source (global, camera-frame). Because the camera is a fixed identity gauge and the Gaussians are free parameters, the only hard requirements are internal self-consistency (a fixed bijection direction→pixel), elevation using the asin/horizon-centred convention, and azimuth wrap — all satisfied here. The chosen `P` makes elevation physically meaningful; the empirical alignment check (§9) confirms the renderer reproduces this mapping.

> Note: most energy sits near the horizon (`glob_theta_r` mostly 85°–118°), which under the horizon-centred map gives occupied rows ≈**30–47** (85°→row 47.0, 118°→row 30.5). We keep the full native 90-row elevation axis for MVP (matches the renderer; no cropping).

### 5.3 Densification into a beamformed-style map
The original GT is **dense** (≈100% nonzero pixels, broad main-lobes spanning 900–2900 px, every image saturates to 255). To match the renderer/SSIM/densification inductive bias, render each path as a **broad anisotropic lobe** (a synthetic beampattern main-lobe centred at the path direction) over a small constant floor. **Order of operations (fixed):**
```
lobes(r,c) = Σ_k  p_k · exp( −0.5·[ ((Δrow)/σ_el)^2 + ((Δcol_wrapped)/σ_az)^2 ] )
S(r,c)     = FLOOR_ABS · max(lobes) + lobes        # floor is a fraction of the pre-floor lobe peak
# then per-image peak normalize in §5.4
```
- `Δcol_wrapped` respects the 360-column azimuth wrap; `Δrow` is plain (no wrap).
- **Axis-to-degree scale:** 2°/row elevation (180° over 90 rows), 1°/col azimuth (360° over 360 cols).
- **Concrete starting constants** (seeds for §9 calibration, not hard constraints): `σ_az = 8 cols (≈8° azimuth)`, `σ_el = 8 rows (≈16° elevation)` — elevation broader because the ULA does not resolve it — and `FLOOR_ABS = 0.03` (3% of the lobe peak) so nonzero-fraction → ~1.0. The array-beamwidth analogy is motivation only; the final values come from §9 calibration.
- **Same-bin collisions** (≈56% of path pairs are within 2° azimuth): combine by **power sum** (incoherent superposition), which the Σ above already does.

### 5.4 Normalization → PNG
**Per-image peak normalization** (matches the verified original convention — every reference PNG has max==255):
```
S ← S / max(S)            # → [0,1], peak at 1.0
PNG = round(S · 255).astype(uint8)     # 90×360, mode 'L'
```
(We choose per-image peak over a fixed global-dB floor because the originals are per-image peak-normalized; a global scheme would make most CAV frames systematically dimmer. The underlying NeRF2 generator's exact scheme is unconfirmed, so this is validated empirically by comparing brightness stats, §9.)

### 5.5 Calibration targets & acceptance (for §9 tuning)
Tune `σ_az, σ_el, FLOOR_ABS` so the synthesized images' brightness statistics, **averaged over ≥50 held-out synthesized samples**, match the reference distribution within tolerance:

| Statistic | Target (reference) | Accept if mean over ≥50 imgs within |
|---|---|---|
| nonzero-fraction | ≈1.0 | ≥ 0.98 |
| per-image mean-norm | ≈0.40 (ref range 0.29–0.58) | 0.30 – 0.55 |
| fraction of pixels > 0.5 | ≈0.32 | 0.20 – 0.45 |

All three must hold simultaneously. There are 3 free knobs for 3 targets, so a small grid/coordinate search suffices.

---

## 6. Coordinate frame, gateway, and `tx_pos`

- **Gateway gauge.** `tx_pos` only conditions the MLP (verified), so the absolute gateway pose is a free gauge. Author the new `gateway_info.yml` with an **identity gauge** consistent with §5.2. The file **must** be nested under a `gateway1:` key (read at `scene/__init__.py:44-45`) with scipy **xyzw** quaternion ordering:
  ```yaml
  dataset_name: <town>_<scenario>_<cav-set>
  gateway1:
    position: [0.0, 0.0, 0.0]
    orientation: [0.0, 0.0, 0.0, 1.0]   # scipy xyzw identity (NOT [1,0,0,0])
  ```
  (Reference `data_test200/gateway_info.yml` is non-identity — position `[5,0.26,0]`, quaternion `[0.513,-0.487,-0.487,0.513]`. Authoring identity is permissible **only because** §5.2 synthesizes the GT in the renderer's own axis convention, keeping synthesis and render frame self-consistent.)

- **`tx_pos` transform.** `tx_pos = R_rsu⁻¹ · (cav_xyz − rsu_xyz) / scale`.
  - `R_rsu⁻¹` (RSU yaw) is **optional/cosmetic** for correctness (any fixed invertible linear map is absorbed by the MLP) but kept for interpretability.
  - **`scale` is measured, not assumed:** compute `scale = max ||cav_xyz − rsu_xyz||` over the chosen (scenario, RSU) split from the pose files (worked scenario: max ≈ 75.5 m); use this single fixed constant (per-dataset, never per-axis/per-sample) so normalized magnitude stays ≲1.0 — comfortably inside the `multires=6` (max-freq 32) encoding's non-aliasing range. (Reference RFID norms reached ~1.33; staying ≤1.0 is safe.)
  - Keep a non-zero `z` offset — the reference `tx_pos` per-axis `z` ∈ [0.86,1.07], not centred at 0. (Note: §2.5's "norm 0.86–1.33" is the **vector norm** range; this [0.86,1.07] is the **z-component** range — different statistics that happen to share a lower endpoint.)
  - **Cull-margin check (tight):** ensure normalized `||tx_pos|| > 0.04`-equivalent of the `too_close()` `rr<=0.04` radius. Measured min CAV–RSU distance is **~15.3 m** → normalized min `||tx_pos|| ≈ 0.203` with `scale=75.5`, which clears the cull radius but with little margin. **Assert `min ||tx_pos|| > 0.2` and re-verify per scenario** (some scenarios may have closer passes).

---

## 7. Output dataset format (converter target)

The converter writes a self-contained directory **format-identical to `data_test200/`**:
```
<out_dir>/
  gateway_info.yml          # nested gateway1:, identity gauge (§6)
  tx_pos.csv                # header x,y,z; ONE ROW PER SAMPLE, ordered so row index = (filename_number − 1)
  spectrum/NNNNN.png        # 90×360 uint8 mode 'L' (§5); ID zero-padded to 5 digits
  train_index.txt           # sample IDs, 80% (converter-written, §4)
  test_index.txt            # 20% (converter-written, §4)
```
**Deterministic enumeration (pins the off-by-one):** sort samples by `(cav_index, frame)` ascending; the i-th sample (0-based) gets **1-based ID `NNNNN = i + 1`**, zero-padded to **5 digits** (distinct from the 6-digit source frame number). In a single pass per sample, write `spectrum/NNNNN.png` **and** append its row to `tx_pos.csv` so that the PNG `NNNNN` always lands at `tx_pos.csv` row `NNNNN − 1` **by construction** — matching the loader's 1-based read `tx_pos[int(index) − 1]` (`scene/dataloader.py:58`).

---

## 8. Code changes (minimal, located)

1. **NEW** `tools/convert_multimodal_to_wrfgs.py` — CLI takes `--weather --town --scenario --antenna-config --rsu --out-dir` + output dir; performs the §4–§7 conversion; prints the measured `scale`, sample count, and brightness stats. To resolve the Channel-vs-Sensor seed asymmetry (§3), it globs `Sensor Data/<Town>/<scenario>_seed*` and **asserts exactly one match** (else require an explicit `--sensor-dir` override). Self-contained; no model code touched.
2. **`scene/__init__.py:38`** — `self.datadir` is **hardcoded** to `'./data_test200'` and overrides the dataset arg. Make it configurable by threading a new **`--datadir`** arg through `ModelParams`/args into `Scene` (chosen over reusing `source_path`, which carries 3DGS COLMAP semantics). (Also remove the dead `datadir='data'` at `train.py:59`.)
3. **No changes** to the renderer, `GaussianModel`, `DeformNetwork`, or the training/eval loops. The dataset class `Spectrum_dataset` (registered as `'rfid'`) is reused as-is.
4. Optionally expose `iterations` / densification knobs for the smaller sample count (tune, not restructure).

---

## 9. Validation plan (evidence before training at scale)

1. **Render↔target alignment check (must pass first).** Render a single known Gaussian and confirm its pixel lands where §5.2 predicts; sweep a few directions to confirm the row (elevation, horizon-centred) and azimuth-wrap conventions empirically — the pixel↔angle map lives in CUDA, so verify, don't assume.
2. **Synthesis sanity.** Visualize several synthesized GT spectra; confirm lobes appear at the expected arrival directions and that brightness stats match §5.5 targets. Adjust `σ_az, σ_el, floor_level`.
3. **Overfit a tiny subset.** Confirm loss decreases and rendered ≈ GT on a handful of samples.
4. **Density ablation (settles the one untested inference).** Train the dense target vs a sparse-blob variant on a small subset; confirm the dense target converges. (Density mismatch is a *verified fact*; the convergence consequence is a hypothesis to confirm here.)
5. **Full per-scene training.** Report median SSIM and mean pixel error (existing eval, `train.py:237-288`), plus a **power-weighted angular error** (PWAE): for each image, the magnitude-weighted mean angular distance between predicted and GT pixels, computed as `Σ_p w_p · d_ang(argmax-aligned)` — concretely, take the GT and predicted peak (or top-k lobes), weight each by its power, and report the mean azimuth+elevation offset in degrees. This is a sparsity-robust companion to SSIM.
6. **Baseline.** Compare against a trivial predictor (mean spectrum over the train set) to confirm the model learns genuine `tx_pos` dependence.

---

## 10. Risks & open items

- **Density realism is synthetic.** Our lobes are a synthetic beampattern, not a physically-measured dense map (the 16-ULA can't resolve elevation). Calibrated to the original's statistics; acceptable for the MVP's goal (demonstrate WRF-GS+ reconstructs a position-conditioned 2D spectrum).
- **Elevation is weakly observed** (paths near the horizon; ULA unresolved in elevation). Full 90-row axis retained for compatibility; most signal occupies a central band.
- **`scale` and brightness calibration are per-scene** — recompute when changing scenario/RSU.
- **Per-scene model** — one model per (scenario, RSU); the converter is parameterized to regenerate per scene.
- **Normalization scheme** of the original NeRF2 generator is unconfirmed; per-image peak chosen on strong indirect evidence and validated by stats matching.

---

## 11. Definition of done (MVP)

- Converter produces a `data_test200`-format dataset for the worked scenario with passing alignment + synthesis-stats checks.
- `datadir` is configurable; `python train.py` trains on the converted dataset with the model/renderer unchanged.
- A full per-scene run reports median SSIM + mean pixel error + power-weighted angular error, beating the mean-spectrum baseline.
- The converter is parameterized to regenerate any other single V2I `sunny` scenario/RSU without code edits.
