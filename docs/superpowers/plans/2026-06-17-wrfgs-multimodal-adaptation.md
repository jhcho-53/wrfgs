# WRF-GS+ → Multimodal-Wireless (V2I) Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Multimodal-Wireless V2I dataset into a `data_test200`-format dataset of densified 2D DoA spatial-spectra and train the unchanged WRF-GS+ model on it (one model per RSU/scenario).

**Architecture:** A standalone converter package `tools/mw2wrfgs/` turns Sionna ray-tracing paths + CARLA poses into `spectrum/*.png` + `tx_pos.csv` + `gateway_info.yml` + split files. The renderer/model are untouched except a one-line change making the dataset directory configurable. Pure geometry/synthesis functions are built test-first; GPU-dependent alignment/training are validated by scripts with documented expected output.

**Tech Stack:** Python 3.8 (conda env `wrfgsplus`), numpy, scipy (`Rotation`), pyyaml, pandas, imageio, pytest; the repo's custom equirectangular `diff-gaussian-rasterization`.

**Spec:** [docs/superpowers/specs/2026-06-17-wrfgs-multimodal-wireless-adaptation-design.md](../specs/2026-06-17-wrfgs-multimodal-wireless-adaptation-design.md). Section refs (§N) below point there.

**Environment note:** Run every command after `conda activate wrfgsplus`. `imageio` comes from `environment.yml`. The dataset root is `/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless`; the worked example is `weather=sunny, antenna=Nt_1_64_Nr_1_16_fc_28GHz, town=Town05, scenario=Town05_parkinglot, rsu=rsu_1`.

---

## File structure

```
tools/mw2wrfgs/
  __init__.py        # package marker + version-free re-exports
  geometry.py        # angles_to_pixel(), build_tx_pos()  (pure numpy/scipy)
  spectrum.py        # per_path_power(), synthesize_spectrum(), spectrum_to_uint8()
  dataio.py          # path/pose loaders + seed-dir resolution + PNG save
  assemble.py        # enumerate_samples(), measure_scale(), build_spectrum_for_sample(), write_dataset()
  cli.py             # argparse entrypoint orchestrating a full conversion
tools/check_alignment.py     # GPU: validate renderer lon/lat projection vs our formula (§9.1)
tools/calibrate_spectrum.py  # tune sigma/floor to reference brightness stats (§5.5/§9.2)
tests/
  conftest.py        # puts repo root + tools/ on sys.path
  test_geometry.py
  test_spectrum.py
  test_dataio.py
  test_assemble.py
  test_arguments_datadir.py
```

Modified existing files: `arguments/__init__.py` (add `datadir` field), `scene/__init__.py:38` (use `args.datadir`), `train.py:59` (remove dead var).

---

## Task 0: Test scaffolding

**Files:**
- Create: `tests/conftest.py`
- Create: `tools/mw2wrfgs/__init__.py`

- [ ] **Step 1: Create the package marker**

`tools/mw2wrfgs/__init__.py`:
```python
"""Converter from Multimodal-Wireless (V2I) to WRF-GS+ data_test200 format."""
```

- [ ] **Step 2: Create conftest so tests can import the package and repo modules**

`tests/conftest.py`:
```python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                      # for `import arguments`, `import scene`
sys.path.insert(0, os.path.join(ROOT, "tools"))  # for `import mw2wrfgs.*`
```

- [ ] **Step 3: Verify pytest discovers an empty suite**

Run: `cd /home/hanyan_arch/jaehyeon/WRF-GSplus && python -m pytest tests/ -q`
Expected: `no tests ran` (exit code 5) — confirms collection works with no errors.

- [ ] **Step 4: Commit**

```bash
git add tools/mw2wrfgs/__init__.py tests/conftest.py
git commit -m "test: scaffold mw2wrfgs package and pytest path setup"
```

---

## Task 1: Geometry — angles → pixel, and tx_pos

**Files:**
- Create: `tools/mw2wrfgs/geometry.py`
- Test: `tests/test_geometry.py`

These are the §5.2 / §6 formulas. `angles_to_pixel` mirrors the renderer's equirectangular map (`lon=atan2(x,z)`, `lat=asin(y/r)`, `ndc2Pix`) with the fixed axis permutation that makes world-up (z) the elevation axis.

- [ ] **Step 1: Write the failing tests**

`tests/test_geometry.py`:
```python
import numpy as np
import pytest
import mw2wrfgs.geometry as geo


def test_horizon_maps_to_image_centre_row():
    # zenith=90deg (horizon), azimuth=0 -> row ~44.5 (centre), col ~269.5
    row, col = geo.angles_to_pixel(np.pi / 2, 0.0)
    assert row == pytest.approx(44.5, abs=1e-6)
    assert col == pytest.approx(269.5, abs=1e-6)


def test_azimuth_half_pi_is_centre_column():
    row, col = geo.angles_to_pixel(np.pi / 2, np.pi / 2)
    assert row == pytest.approx(44.5, abs=1e-6)
    assert col == pytest.approx(179.5, abs=1e-6)


def test_zenith_maps_to_top_row():
    # straight up (zenith=0) -> row ~89.5 (top of image)
    row, _ = geo.angles_to_pixel(0.0, 1.234)
    assert row == pytest.approx(89.5, abs=1e-6)


def test_known_elevation_band():
    # spec note: 85deg -> row 47.0, 118deg -> row 30.5
    r85, _ = geo.angles_to_pixel(np.deg2rad(85.0), 0.0)
    r118, _ = geo.angles_to_pixel(np.deg2rad(118.0), 0.0)
    assert r85 == pytest.approx(47.0, abs=0.1)
    assert r118 == pytest.approx(30.5, abs=0.1)


def test_angles_to_pixel_is_vectorized():
    th = np.array([np.pi / 2, 0.0])
    ph = np.array([0.0, 1.234])
    row, col = geo.angles_to_pixel(th, ph)
    assert row.shape == (2,)
    assert row[0] == pytest.approx(44.5, abs=1e-6)


def test_build_tx_pos_identity_yaw():
    tx = geo.build_tx_pos([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 10.0)
    np.testing.assert_allclose(tx, [1.0, 0.0, 0.0], atol=1e-9)


def test_build_tx_pos_rotates_into_rsu_local_frame():
    # yaw=90deg: inverse rotation sends world +x to -y
    tx = geo.build_tx_pos([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 90.0, 10.0)
    np.testing.assert_allclose(tx, [0.0, -1.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_geometry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mw2wrfgs.geometry'`.

- [ ] **Step 3: Implement `geometry.py`**

`tools/mw2wrfgs/geometry.py`:
```python
import numpy as np
from scipy.spatial.transform import Rotation

IMG_H = 90
IMG_W = 360


def _ndc2pix(v, size):
    # mirrors auxiliary.h ndc2Pix: ((v + 1) * size - 1) / 2
    return ((v + 1.0) * size - 1.0) / 2.0


def angles_to_pixel(glob_theta, glob_phi):
    """Map Sionna world-frame arrival angles to renderer (row, col) pixel centres.

    glob_theta: zenith angle in [0, pi]. glob_phi: azimuth in [-pi, pi].
    Scalars or numpy arrays. Returns (row, col) as float(s).

    Mirrors the equirectangular rasterizer (forward.cu / auxiliary.h):
    lon = atan2(x_cam, z_cam), lat = asin(y_cam / r), then ndc2Pix.
    Axis permutation P makes world-up (+z) the renderer's elevation axis (+y).
    """
    theta = np.asarray(glob_theta, dtype=np.float64)
    phi = np.asarray(glob_phi, dtype=np.float64)

    # world unit direction of arrival (Sionna physics convention, z-up)
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    # P: camera_x = world_x, camera_y = world_z (up), camera_z = world_y
    cx, cy, cz = dx, dz, dy

    lon = np.arctan2(cx, cz)                     # [-pi, pi]
    lat = np.arcsin(np.clip(cy, -1.0, 1.0))      # [-pi/2, pi/2] (unit vector)

    col = _ndc2pix(lon / np.pi, IMG_W)
    row = _ndc2pix(lat * 2.0 / np.pi, IMG_H)
    return row, col


def build_tx_pos(cav_xyz, rsu_xyz, rsu_yaw_deg, scale):
    """RSU-local, scaled conditioning position (spec §6).

    tx = R_rsu^{-1} . (cav - rsu) / scale, R_rsu = yaw rotation about world z.
    Returns a (3,) float64 array.
    """
    cav = np.asarray(cav_xyz, dtype=np.float64)
    rsu = np.asarray(rsu_xyz, dtype=np.float64)
    rel = cav - rsu
    r_inv = Rotation.from_euler("z", rsu_yaw_deg, degrees=True).inv()
    return r_inv.apply(rel) / float(scale)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_geometry.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/mw2wrfgs/geometry.py tests/test_geometry.py
git commit -m "feat: equirectangular angles_to_pixel and rsu-local tx_pos (geometry)"
```

---

## Task 2: Spectrum synthesis — per-path power, densified lobes, uint8

**Files:**
- Create: `tools/mw2wrfgs/spectrum.py`
- Test: `tests/test_spectrum.py`

Implements §5.1, §5.3, §5.4. `synthesize_spectrum` builds the dense beamformed-style image; `spectrum_to_uint8` does per-image peak normalization already applied inside `synthesize_spectrum`, then maps to bytes.

- [ ] **Step 1: Write the failing tests**

`tests/test_spectrum.py`:
```python
import numpy as np
import pytest
import mw2wrfgs.spectrum as spec


def _make_a(powers):
    """Build a Sionna-shaped `a` (1,1,Nr,1,Nt,n_paths,1) whose per-element
    magnitude per path equals sqrt(power) (antenna dims are phase-only)."""
    n = len(powers)
    a = np.zeros((1, 1, 2, 1, 3, n, 1), dtype=np.complex64)
    for k, p in enumerate(powers):
        a[:, :, :, :, :, k, :] = np.sqrt(p)  # constant across antenna dims
    return a


def test_per_path_power_uses_element_zero_magnitude_squared():
    a = _make_a([25.0, 1.0])
    p = spec.per_path_power(a)
    np.testing.assert_allclose(p, [25.0, 1.0], rtol=1e-5)


def test_per_path_power_zeros_padded_slots():
    a = _make_a([4.0, 0.0])
    p = spec.per_path_power(a)
    np.testing.assert_allclose(p, [4.0, 0.0], rtol=1e-5)


def test_synthesize_peak_at_path_centre_and_dense():
    S = spec.synthesize_spectrum(
        np.array([1.0]), np.array([45.0]), np.array([180.0]),
        sigma_az=8.0, sigma_el=8.0, floor_abs=0.03,
    )
    assert S.shape == (90, 360)
    assert S.max() == pytest.approx(1.0, abs=1e-9)
    # peak is at the path centre
    assert np.unravel_index(np.argmax(S), S.shape) == (45, 180)
    # dense: every pixel > 0 because of the floor
    assert (S > 0).mean() == pytest.approx(1.0, abs=1e-9)
    # floor level ~= floor_abs / (1 + floor_abs)
    assert S.min() == pytest.approx(0.03 / 1.03, abs=1e-3)


def test_synthesize_azimuth_wraps():
    # a path at col 0 should put energy near col 359 too (wrap), more than at col 180
    S = spec.synthesize_spectrum(
        np.array([1.0]), np.array([45.0]), np.array([0.0]),
        sigma_az=8.0, sigma_el=8.0, floor_abs=0.03,
    )
    assert S[45, 359] > S[45, 180]


def test_synthesize_empty_returns_zeros():
    S = spec.synthesize_spectrum(np.array([0.0]), np.array([45.0]), np.array([180.0]))
    assert S.shape == (90, 360)
    assert S.max() == 0.0


def test_spectrum_to_uint8_peak_is_255():
    S = np.zeros((90, 360))
    S[10, 20] = 1.0
    img = spec.spectrum_to_uint8(S)
    assert img.dtype == np.uint8
    assert img[10, 20] == 255
    assert img.shape == (90, 360)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_spectrum.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mw2wrfgs.spectrum'`.

- [ ] **Step 3: Implement `spectrum.py`**

`tools/mw2wrfgs/spectrum.py`:
```python
import numpy as np

IMG_H = 90
IMG_W = 360


def per_path_power(a):
    """Intrinsic per-path power from Sionna gains (spec §5.1).

    a: complex array (1, 1, Nr, 1, Nt, n_paths, 1). Antenna dims carry only
    phase, so |a| is constant across them; element [0,0,0,0,0,k,0] suffices.
    Returns a real (n_paths,) array; zero-|a| (padded) slots come out 0.
    """
    a0 = a[0, 0, 0, 0, 0, :, 0]
    return np.abs(a0).astype(np.float64) ** 2


def synthesize_spectrum(p, row, col, sigma_az=8.0, sigma_el=8.0, floor_abs=0.03):
    """Densified 2D DoA spectrum (spec §5.3/§5.4).

    p, row, col: (n_paths,) per-path power and renderer pixel centres.
    Returns a (90, 360) float64 image in [0, 1], per-image peak-normalized,
    or all-zeros if there is no positive power.
    """
    p = np.asarray(p, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64)

    rr = np.arange(IMG_H, dtype=np.float64).reshape(IMG_H, 1)
    cc = np.arange(IMG_W, dtype=np.float64).reshape(1, IMG_W)

    lobes = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    for pk, r0, c0 in zip(p, row, col):
        if pk <= 0:
            continue
        d_row = rr - r0                                   # (H, 1)
        d_col = cc - c0                                   # (1, W)
        d_col = (d_col + IMG_W / 2.0) % IMG_W - IMG_W / 2.0  # azimuth wrap
        g = np.exp(-0.5 * ((d_row / sigma_el) ** 2 + (d_col / sigma_az) ** 2))
        lobes = lobes + pk * g

    peak = lobes.max()
    if peak <= 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.float64)

    S = floor_abs * peak + lobes          # constant floor as a fraction of the lobe peak
    return S / S.max()                    # per-image peak normalization -> [0, 1]


def spectrum_to_uint8(S):
    """Map a [0,1] spectrum to an 8-bit grayscale array (90, 360)."""
    return np.clip(np.round(np.asarray(S) * 255.0), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_spectrum.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/mw2wrfgs/spectrum.py tests/test_spectrum.py
git commit -m "feat: densified 2D DoA spectrum synthesis (spectrum)"
```

---

## Task 3: Data IO — path/pose loaders, seed-dir resolution, PNG save

**Files:**
- Create: `tools/mw2wrfgs/dataio.py`
- Test: `tests/test_dataio.py`

Implements the §3 readers, the §8 seed-asymmetry resolution, and the PNG writer (the only `imageio` user).

- [ ] **Step 1: Write the failing tests**

`tests/test_dataio.py`:
```python
import os
import numpy as np
import pytest
import mw2wrfgs.dataio as dio


def test_resolve_sensor_dir_requires_exactly_one(tmp_path):
    root = tmp_path
    base = root / "sunny" / "Sensor Data" / "Town05"
    (base / "Town05_parkinglot_seed42").mkdir(parents=True)
    got = dio.resolve_sensor_dir(str(root), "sunny", "Town05", "Town05_parkinglot")
    assert got.endswith("Town05_parkinglot_seed42")

    # a second seed match -> error
    (base / "Town05_parkinglot_seed7").mkdir()
    with pytest.raises(ValueError):
        dio.resolve_sensor_dir(str(root), "sunny", "Town05", "Town05_parkinglot")


def test_load_paths_returns_a_and_global_angles(tmp_path):
    f = tmp_path / "000280_paths.npz"
    a = np.zeros((1, 1, 2, 1, 3, 2, 1), dtype=np.complex64)
    a[:] = 1.0
    np.savez(
        f, a=a,
        glob_theta_r=np.array([[[[1.5, 1.7]]]], dtype=np.float32),
        glob_phi_r=np.array([[[[0.3, -0.4]]]], dtype=np.float32),
    )
    aa, th, ph = dio.load_paths(str(f))
    assert aa.shape == (1, 1, 2, 1, 3, 2, 1)
    np.testing.assert_allclose(th.ravel(), [1.5, 1.7], rtol=1e-5)
    np.testing.assert_allclose(ph.ravel(), [0.3, -0.4], rtol=1e-5)


def test_load_cav_position_reads_sensors_vehicle_pose(tmp_path):
    f = tmp_path / "000480.yaml"
    f.write_text(
        "actor: cav_1\nframe: 480\n"
        "sensors:\n"
        "  vehicle_pose:\n"
        "    location: {x: -30.35, y: -4.38, z: 0.002}\n"
    )
    pos = dio.load_cav_position(str(f))
    np.testing.assert_allclose(pos, [-30.35, -4.38, 0.002], atol=1e-6)


def test_load_rsu_pose_reads_config(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(
        "scenarios:\n"
        "  Town05_parkinglot:\n"
        "    rsu_transform:\n"
        "      location: {x: -59, y: 11, z: 0}\n"
        "      rotation: {yaw: 45}\n"
    )
    loc, yaw = dio.load_rsu_pose(str(f), "Town05_parkinglot")
    np.testing.assert_allclose(loc, [-59, 11, 0], atol=1e-6)
    assert yaw == pytest.approx(45.0)


def test_save_spectrum_png_roundtrips_shape(tmp_path):
    import imageio.v2 as imageio
    arr = np.zeros((90, 360), dtype=np.uint8)
    arr[10, 20] = 255
    out = tmp_path / "00001.png"
    dio.save_spectrum_png(str(out), arr)
    back = imageio.imread(str(out))
    assert back.shape == (90, 360)
    assert back[10, 20] == 255
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dataio.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mw2wrfgs.dataio'`.

- [ ] **Step 3: Implement `dataio.py`**

`tools/mw2wrfgs/dataio.py`:
```python
import glob
import os

import imageio.v2 as imageio
import numpy as np
import yaml


def channel_scenario_dir(root, weather, antenna_config, town, scenario):
    """Channel Data path (NO _seed suffix on scenario)."""
    return os.path.join(root, weather, "Channel Data", "V2I",
                        antenna_config, town, scenario)


def resolve_sensor_dir(root, weather, town, scenario):
    """Sensor Data scenario dir carries a _seed* suffix; require exactly one match."""
    pattern = os.path.join(root, weather, "Sensor Data", town, scenario + "_seed*")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one sensor dir for {!r}, got {}".format(pattern, matches)
        )
    return matches[0]


def load_paths(npz_path):
    """Return (a, glob_theta_r, glob_phi_r) from a Sionna *_paths.npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return d["a"], d["glob_theta_r"], d["glob_phi_r"]


def load_cav_position(yaml_path):
    """Ground-truth CAV location at sensors.vehicle_pose.location -> (3,) array."""
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)
    loc = y["sensors"]["vehicle_pose"]["location"]
    return np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)


def load_rsu_pose(config_yaml_path, scenario):
    """Return (rsu_location (3,), rsu_yaw_deg) from a scenario config.yaml."""
    with open(config_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    t = cfg["scenarios"][scenario]["rsu_transform"]
    loc = t["location"]
    yaw = float(t["rotation"]["yaw"])
    return np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64), yaw


def save_spectrum_png(path, uint8_img):
    """Write a (90,360) uint8 grayscale PNG (mode 'L'), matching data_test200."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, uint8_img)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_dataio.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/mw2wrfgs/dataio.py tests/test_dataio.py
git commit -m "feat: dataset readers, seed-dir resolution, png writer (dataio)"
```

---

## Task 4: Assembly — enumerate, measure scale, per-sample spectrum, write dataset

**Files:**
- Create: `tools/mw2wrfgs/assemble.py`
- Test: `tests/test_assemble.py`

Implements §4 (enumeration, split ownership), §6 (scale measurement + cull margin), §7 (output layout, off-by-one). `write_dataset` is the orchestrator.

- [ ] **Step 1: Write the failing tests**

`tests/test_assemble.py`:
```python
import os
import numpy as np
import pytest
import imageio.v2 as imageio
import yaml
import mw2wrfgs.assemble as asm


def _write_npz(path, thetas, phis, powers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(powers)
    a = np.zeros((1, 1, 2, 1, 3, n, 1), dtype=np.complex64)
    for k, p in enumerate(powers):
        a[:, :, :, :, :, k, :] = np.sqrt(p)
    np.savez(path, a=a,
             glob_theta_r=np.array(thetas, dtype=np.float32).reshape(1, 1, 1, n),
             glob_phi_r=np.array(phis, dtype=np.float32).reshape(1, 1, 1, n))


def _write_pose(path, x, y, z):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"sensors": {"vehicle_pose": {"location": {"x": x, "y": y, "z": z}}}}, f)


def _build_fake_scene(tmp_path):
    ch = tmp_path / "ch"        # channel scenario dir
    sen = tmp_path / "sen"      # sensor dir
    frames = [280, 281]
    for cav, cav_x in (("cav_1", -20.0), ("cav_2", -30.0)):
        for fr in frames:
            _write_npz(str(ch / cav / "{:06d}_paths.npz".format(fr)),
                       thetas=[np.pi / 2], phis=[0.0], powers=[1.0])
            _write_pose(str(sen / cav / "{:06d}.yaml".format(fr)), cav_x, 0.0, 0.0)
    return str(ch), str(sen)


def test_enumerate_orders_by_cav_then_frame(tmp_path):
    ch, sen = _build_fake_scene(tmp_path)
    samples = asm.enumerate_samples(ch, sen)
    keys = [(s[0], s[1]) for s in samples]
    assert keys == [(1, 280), (1, 281), (2, 280), (2, 281)]


def test_enumerate_skips_frames_without_pose(tmp_path):
    ch, sen = _build_fake_scene(tmp_path)
    os.remove(os.path.join(sen, "cav_1", "000281.yaml"))
    samples = asm.enumerate_samples(ch, sen)
    keys = [(s[0], s[1]) for s in samples]
    assert (1, 281) not in keys and len(keys) == 3


def test_measure_scale_is_max_distance(tmp_path):
    ch, sen = _build_fake_scene(tmp_path)
    samples = asm.enumerate_samples(ch, sen)
    scale = asm.measure_scale(samples, np.array([0.0, 0.0, 0.0]))
    assert scale == pytest.approx(30.0, abs=1e-6)  # cav_2 at x=-30


def test_write_dataset_layout_and_offbyone(tmp_path):
    ch, sen = _build_fake_scene(tmp_path)
    out = tmp_path / "out"
    samples = asm.enumerate_samples(ch, sen)
    stats = asm.write_dataset(
        str(out), samples,
        rsu_xyz=np.array([0.0, 0.0, 0.0]), rsu_yaw=0.0,
        scale=30.0, sigma_az=8.0, sigma_el=8.0, floor_abs=0.03,
        train_ratio=0.75, seed=0, dataset_name="fake",
    )
    # files exist
    assert os.path.exists(out / "gateway_info.yml")
    assert os.path.exists(out / "tx_pos.csv")
    assert os.path.exists(out / "train_index.txt")
    assert os.path.exists(out / "test_index.txt")
    assert os.path.exists(out / "spectrum" / "00001.png")
    assert os.path.exists(out / "spectrum" / "00004.png")

    # gateway is nested + identity gauge
    gw = yaml.safe_load(open(out / "gateway_info.yml"))
    assert gw["gateway1"]["position"] == [0.0, 0.0, 0.0]
    assert gw["gateway1"]["orientation"] == [0.0, 0.0, 0.0, 1.0]

    # off-by-one: tx_pos.csv has one row per sample, row (ID-1) is cav_1 frame 280
    import pandas as pd
    tx = pd.read_csv(out / "tx_pos.csv").values
    assert tx.shape == (4, 3)
    # sample 1 == cav_1 (x=-20) at rsu origin, scale 30 -> x = -20/30
    np.testing.assert_allclose(tx[0], [-20.0 / 30.0, 0.0, 0.0], atol=1e-6)

    # split sizes (75/25 of 4 -> 3 train, 1 test) and disjoint
    train = set(open(out / "train_index.txt").read().split())
    test = set(open(out / "test_index.txt").read().split())
    assert len(train) == 3 and len(test) == 1
    assert train.isdisjoint(test)
    assert train | test == {"00001", "00002", "00003", "00004"}

    # spectrum png is a valid 90x360 image
    img = imageio.imread(out / "spectrum" / "00001.png")
    assert img.shape == (90, 360)

    # stats surfaced for logging
    assert stats["n_samples"] == 4
    assert stats["min_tx_norm"] > 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_assemble.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mw2wrfgs.assemble'`.

- [ ] **Step 3: Implement `assemble.py`**

`tools/mw2wrfgs/assemble.py`:
```python
import glob
import os

import numpy as np
import pandas as pd
import yaml

from . import dataio, geometry, spectrum


def list_cavs(channel_scenario_dir):
    return sorted(
        d for d in os.listdir(channel_scenario_dir)
        if d.startswith("cav_") and os.path.isdir(os.path.join(channel_scenario_dir, d))
    )


def enumerate_samples(channel_scenario_dir, sensor_dir):
    """List (cav_idx, frame, npz_path, yaml_path) joined on frame, sorted by (cav, frame).

    Only frames present in BOTH channel and sensor trees are kept.
    """
    samples = []
    for cav in list_cavs(channel_scenario_dir):
        cav_idx = int(cav.split("_")[1])
        for npz in glob.glob(os.path.join(channel_scenario_dir, cav, "*_paths.npz")):
            frame = int(os.path.basename(npz).split("_")[0])
            yml = os.path.join(sensor_dir, cav, "{:06d}.yaml".format(frame))
            if os.path.exists(yml):
                samples.append((cav_idx, frame, npz, yml))
    samples.sort(key=lambda s: (s[0], s[1]))
    return samples


def measure_scale(samples, rsu_xyz):
    """Max ||cav - rsu|| over all samples (the single fixed tx_pos scale, §6)."""
    dmax = 0.0
    for (_, _, _, yml) in samples:
        cav = dataio.load_cav_position(yml)
        dmax = max(dmax, float(np.linalg.norm(cav - rsu_xyz)))
    return dmax


def build_spectrum_for_sample(npz_path, sigma_az, sigma_el, floor_abs):
    """Load one path file and synthesize its (90,360) [0,1] DoA spectrum."""
    a, glob_theta_r, glob_phi_r = dataio.load_paths(npz_path)
    p = spectrum.per_path_power(a)
    row, col = geometry.angles_to_pixel(glob_theta_r.ravel(), glob_phi_r.ravel())
    return spectrum.synthesize_spectrum(p, row, col, sigma_az, sigma_el, floor_abs)


def _write_gateway_info(out_dir, dataset_name):
    gw = {
        "dataset_name": dataset_name,
        "gateway1": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],  # scipy xyzw identity
        },
    }
    with open(os.path.join(out_dir, "gateway_info.yml"), "w") as f:
        yaml.safe_dump(gw, f, default_flow_style=None, sort_keys=False)


def _write_split(out_dir, ids, train_ratio, seed):
    rng = np.random.RandomState(seed)
    order = list(ids)
    rng.shuffle(order)
    n_train = int(round(len(order) * train_ratio))
    train = sorted(order[:n_train])
    test = sorted(order[n_train:])
    with open(os.path.join(out_dir, "train_index.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out_dir, "test_index.txt"), "w") as f:
        f.write("\n".join(test) + "\n")
    return len(train), len(test)


def write_dataset(out_dir, samples, rsu_xyz, rsu_yaw, scale,
                  sigma_az, sigma_el, floor_abs,
                  train_ratio=0.8, seed=0, dataset_name="mw_v2i"):
    """Write a data_test200-format dataset. Returns a stats dict.

    Enumeration is fixed by `samples` order (already (cav_idx, frame)-sorted):
    the i-th sample (0-based) -> 1-based 5-digit ID, PNG and tx_pos row written
    in the same pass so file NNNNN.png <-> tx_pos.csv row NNNNN-1 by construction.
    """
    os.makedirs(os.path.join(out_dir, "spectrum"), exist_ok=True)
    rsu_xyz = np.asarray(rsu_xyz, dtype=np.float64)

    ids = []
    tx_rows = []
    min_tx_norm = np.inf
    for i, (_, _, npz, yml) in enumerate(samples):
        sample_id = "{:05d}".format(i + 1)
        ids.append(sample_id)

        S = build_spectrum_for_sample(npz, sigma_az, sigma_el, floor_abs)
        dataio.save_spectrum_png(
            os.path.join(out_dir, "spectrum", sample_id + ".png"),
            spectrum.spectrum_to_uint8(S),
        )

        cav = dataio.load_cav_position(yml)
        tx = geometry.build_tx_pos(cav, rsu_xyz, rsu_yaw, scale)
        tx_rows.append(tx)
        min_tx_norm = min(min_tx_norm, float(np.linalg.norm(tx)))

    pd.DataFrame(np.asarray(tx_rows), columns=["x", "y", "z"]).to_csv(
        os.path.join(out_dir, "tx_pos.csv"), index=False
    )
    _write_gateway_info(out_dir, dataset_name)
    n_train, n_test = _write_split(out_dir, ids, train_ratio, seed)

    return {
        "n_samples": len(samples),
        "scale": float(scale),
        "min_tx_norm": float(min_tx_norm),
        "n_train": n_train,
        "n_test": n_test,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_assemble.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/mw2wrfgs/assemble.py tests/test_assemble.py
git commit -m "feat: sample enumeration, scale, and dataset writer (assemble)"
```

---

## Task 5: CLI entrypoint

**Files:**
- Create: `tools/mw2wrfgs/cli.py`

Orchestrates a full conversion and enforces the §6 cull-margin assertion. No new unit test (logic is covered by Task 4); verified by the real-data run in Task 8.

- [ ] **Step 1: Implement `cli.py`**

`tools/mw2wrfgs/cli.py`:
```python
import argparse
import os

import numpy as np

from . import dataio, assemble


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Multimodal-Wireless V2I -> WRF-GS+ dataset")
    ap.add_argument("--root", default="/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless")
    ap.add_argument("--weather", default="sunny")
    ap.add_argument("--antenna-config", default="Nt_1_64_Nr_1_16_fc_28GHz")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--scenario", default="Town05_parkinglot")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sensor-dir", default=None, help="override seed-glob resolution")
    ap.add_argument("--sigma-az", type=float, default=8.0)
    ap.add_argument("--sigma-el", type=float, default=8.0)
    ap.add_argument("--floor-abs", type=float, default=0.03)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tx-norm", type=float, default=0.2,
                    help="assert min ||tx_pos|| exceeds this (cull-radius margin, §6)")
    args = ap.parse_args(argv)

    ch = dataio.channel_scenario_dir(args.root, args.weather, args.antenna_config,
                                     args.town, args.scenario)
    sen = args.sensor_dir or dataio.resolve_sensor_dir(
        args.root, args.weather, args.town, args.scenario)
    config_yaml = os.path.join(sen, "config.yaml")
    rsu_xyz, rsu_yaw = dataio.load_rsu_pose(config_yaml, args.scenario)

    samples = assemble.enumerate_samples(ch, sen)
    if not samples:
        raise SystemExit("no (cav, frame) samples found under {}".format(ch))
    scale = assemble.measure_scale(samples, rsu_xyz)
    print("[mw2wrfgs] samples={}  rsu={}  yaw={}  scale={:.3f}".format(
        len(samples), rsu_xyz.tolist(), rsu_yaw, scale))

    stats = assemble.write_dataset(
        args.out_dir, samples, rsu_xyz, rsu_yaw, scale,
        args.sigma_az, args.sigma_el, args.floor_abs,
        train_ratio=args.train_ratio, seed=args.seed,
        dataset_name="{}_{}".format(args.town, args.scenario),
    )
    print("[mw2wrfgs] wrote {} -> {}".format(stats, args.out_dir))

    if stats["min_tx_norm"] <= args.min_tx_norm:
        raise SystemExit(
            "min ||tx_pos|| = {:.4f} <= {} : too close to cull radius; "
            "inspect scenario geometry".format(stats["min_tx_norm"], args.min_tx_norm))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the CLI parses and reports help**

Run: `cd /home/hanyan_arch/jaehyeon/WRF-GSplus && PYTHONPATH=tools python -m mw2wrfgs.cli --help`
Expected: argparse usage text listing all flags; exit 0.

- [ ] **Step 3: Commit**

```bash
git add tools/mw2wrfgs/cli.py
git commit -m "feat: mw2wrfgs CLI entrypoint with cull-margin assertion"
```

---

## Task 6: Make the dataset directory configurable (model side)

**Files:**
- Modify: `arguments/__init__.py` (add `datadir` field to `ModelParams`)
- Modify: `scene/__init__.py:38`
- Modify: `train.py:59` (remove dead var)
- Test: `tests/test_arguments_datadir.py`

- [ ] **Step 1: Write the failing test**

`tests/test_arguments_datadir.py`:
```python
from argparse import ArgumentParser
from arguments import ModelParams


def test_datadir_arg_threads_through_extract():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(["--datadir", "/tmp/converted_scene"])
    g = lp.extract(args)
    assert g.datadir == "/tmp/converted_scene"


def test_datadir_has_default():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args([])
    g = lp.extract(args)
    assert g.datadir == "./data_test200"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_arguments_datadir.py -q`
Expected: FAIL — argparse `unrecognized arguments: --datadir` (and/or `g.datadir` AttributeError).

- [ ] **Step 3: Add the `datadir` field to `ModelParams`**

In `arguments/__init__.py`, inside `ModelParams.__init__`, add the field (a non-underscore name auto-creates `--datadir`). Place it right after `self.eval = False`:

```python
        self.eval = False
        self.datadir = "./data_test200"
        super().__init__(parser, "Loading Parameters", sentinel)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_arguments_datadir.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Use the arg in `Scene` and remove the dead var**

In `scene/__init__.py:38`, replace:
```python
        self.datadir = "./data_test200" # Choose the dataset directory  
```
with:
```python
        self.datadir = args.datadir  # configurable via --datadir (default ./data_test200)
```

In `train.py:59`, delete the dead line:
```python
    datadir = 'data'
```

- [ ] **Step 6: Verify nothing else references the removed name and tests still pass**

Run: `grep -n "datadir = 'data'" train.py` → Expected: no output.
Run: `python -m pytest tests/test_arguments_datadir.py -q` → Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add arguments/__init__.py scene/__init__.py train.py tests/test_arguments_datadir.py
git commit -m "feat: make Scene dataset directory configurable via --datadir"
```

---

## Task 7: Renderer projection alignment check (GPU validation, §9.1)

**Files:**
- Create: `tools/check_alignment.py`

Validates that the CUDA rasterizer's `lon=atan2(x,z)`, `lat=asin(y/r)`, `ndc2Pix` projection matches our `angles_to_pixel` formula end-to-end. Requires a GPU and the built `diff-gaussian-rasterization`.

- [ ] **Step 1: Implement the alignment script**

`tools/check_alignment.py`:
```python
"""GPU check: a single Gaussian placed at camera-frame direction v must render
its brightest pixel at the renderer's lon/lat pixel for v. Confirms the CUDA
equirectangular projection matches tools/mw2wrfgs/geometry. Run inside wrfgsplus."""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.generate_camera import generate_new_cam


def renderer_pixel(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    lon = math.atan2(v[0], v[2])
    lat = math.asin(v[1])
    col = ((lon / math.pi + 1.0) * 360 - 1.0) / 2.0
    row = ((lat * 2.0 / math.pi + 1.0) * 90 - 1.0) / 2.0
    return row, col


def render_single(v):
    cam = generate_new_cam(np.eye(3), np.zeros(3))  # identity gateway, 360x90, FoV180
    means3D = torch.tensor([[v[0], v[1], v[2]]], dtype=torch.float32, device="cuda") * 5.0
    means2D = torch.zeros_like(means3D, requires_grad=True)
    opacities = torch.ones((1, 1), device="cuda")
    scales = torch.full((1, 3), 0.05, device="cuda")
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    colors = torch.ones((1, 3), device="cuda")  # colors_precomp -> bright

    rs = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=0, campos=cam.camera_center, prefiltered=False,
        debug=False, antialiasing=False)
    rasterizer = GaussianRasterizer(raster_settings=rs)
    img, _, _ = rasterizer(means3D=means3D, means2D=means2D, shs=None,
                           colors_precomp=colors, opacities=opacities,
                           scales=scales, rotations=rotations, cov3D_precomp=None)
    return img[0].detach().cpu().numpy()  # channel 0, (90, 360)


def main():
    # test directions spanning azimuth and elevation
    dirs = [
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0),
        (0.0, 0.5, 1.0), (0.3, -0.4, 1.0),
    ]
    ok = True
    for v in dirs:
        img = render_single(v)
        r_obs, c_obs = np.unravel_index(int(np.argmax(img)), img.shape)
        r_exp, c_exp = renderer_pixel(v)
        # azimuth wrap-aware column distance
        dc = min(abs(c_obs - c_exp), 360 - abs(c_obs - c_exp))
        dr = abs(r_obs - r_exp)
        good = dr <= 2 and dc <= 2
        ok = ok and good
        print("v={} expected=({:.1f},{:.1f}) observed=({},{}) dr={:.1f} dc={:.1f} {}".format(
            v, r_exp, c_exp, r_obs, c_obs, dr, dc, "OK" if good else "FAIL"))
    print("ALIGNMENT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the alignment check on a GPU**

Run: `cd /home/hanyan_arch/jaehyeon/WRF-GSplus && python tools/check_alignment.py`
Expected: every line ends `OK` and the last line is `ALIGNMENT PASS`. If any line FAILs, the CUDA projection differs from our assumption — STOP and reconcile `geometry.angles_to_pixel` / `renderer_pixel` with `auxiliary.h` before proceeding (do not train on a misaligned target).

- [ ] **Step 3: Commit**

```bash
git add tools/check_alignment.py
git commit -m "test: GPU alignment check for equirectangular projection"
```

---

## Task 8: Convert the worked scenario & inspect synthesis (§9.2)

**Files:**
- Create: `tools/calibrate_spectrum.py`

- [ ] **Step 1: Run the converter on the worked scenario**

Run (from the repo root, with `tools/` on the path so `--out-dir` and `./data_test200` stay relative to the repo):
```bash
cd /home/hanyan_arch/jaehyeon/WRF-GSplus
PYTHONPATH=tools python -m mw2wrfgs.cli \
  --out-dir ./data_mw_town05_parkinglot \
  --town Town05 --scenario Town05_parkinglot \
  2>&1 | tee /tmp/mw_convert.log
```
Expected: prints `samples=3300` (±, exact = available frames), a finite `scale` (~75), `min_tx_norm` > 0.2, and `wrote {...}`. The output dir contains `spectrum/` (one PNG per sample), `tx_pos.csv`, `gateway_info.yml`, `train_index.txt`, `test_index.txt`. No assertion error.

- [ ] **Step 2: Implement the calibration/stats script**

`tools/calibrate_spectrum.py`:
```python
"""Compare synthesized vs reference (data_test200) brightness stats (spec §5.5).
Reports nonzero-fraction, mean-norm, fraction>0.5 over N images and whether they
fall in the acceptance bands. Use it to tune --sigma-az/--sigma-el/--floor-abs."""
import argparse
import glob
import os

import numpy as np
import imageio.v2 as imageio


def stats(folder, n=50):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))[:n]
    nz, mn, hi = [], [], []
    for f in files:
        im = imageio.imread(f).astype(np.float64) / 255.0
        nz.append((im > 0).mean())
        mn.append(im.mean())
        hi.append((im > 0.5).mean())
    return np.mean(nz), np.mean(mn), np.mean(hi), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="converted spectrum/ dir")
    ap.add_argument("--ref", default="./data_test200/spectrum")
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    s = stats(args.synth, args.n)
    r = stats(args.ref, args.n)
    print("synth nonzero={:.3f} mean={:.3f} frac>0.5={:.3f} (n={})".format(*s))
    print("ref   nonzero={:.3f} mean={:.3f} frac>0.5={:.3f} (n={})".format(*r))
    bands = (s[0] >= 0.98, 0.30 <= s[1] <= 0.55, 0.20 <= s[2] <= 0.45)
    print("acceptance (nonzero>=.98, mean in[.30,.55], frac>.5 in[.20,.45]):", bands)
    print("CALIBRATION", "PASS" if all(bands) else "ADJUST sigma/floor")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Check synthesized brightness stats against the reference**

Run: `python tools/calibrate_spectrum.py --synth ./data_mw_town05_parkinglot/spectrum`
Expected: `CALIBRATION PASS`. If `ADJUST`, re-run Task 8 Step 1 with tuned `--sigma-az/--sigma-el/--floor-abs` (raise floor → higher mean/nonzero; widen sigma → larger frac>0.5) until all three bands hold. Also eyeball a few PNGs (e.g. `python -c "import imageio.v2 as i,numpy as np; a=i.imread('data_mw_town05_parkinglot/spectrum/00001.png'); print(a.shape, a.min(), a.max())"`).

- [ ] **Step 4: Commit**

```bash
git add tools/calibrate_spectrum.py
git commit -m "test: brightness calibration script for synthesized spectra"
```

> The converted dataset directory `data_mw_town05_parkinglot/` is a build artifact — do NOT commit it. Add it to `.gitignore` in this step:
> ```bash
> echo "data_mw_*/" >> .gitignore && git add .gitignore && git commit -m "chore: ignore converted mw datasets"
> ```

---

## Task 9: Overfit smoke test (§9.3)

- [ ] **Step 1: Train briefly on the converted dataset**

Run:
```bash
cd /home/hanyan_arch/jaehyeon/WRF-GSplus
python train.py --datadir ./data_mw_town05_parkinglot \
  --iterations 3000 --disable_viewer \
  --test_iterations 3000 --save_iterations 3000 --checkpoint_iterations 3000 \
  2>&1 | tee /tmp/mw_train_smoke.log
```
Expected: training starts, the tqdm `Loss` decreases over iterations (not NaN, not flat at the initial value), and the iteration-3000 evaluation prints per-spectrum SSIM lines and a median SSIM (see `logs/<timestamp>/logger.log`). The TensorBoard `render-img` should visually resemble the GT lobes.

- [ ] **Step 2: Confirm the eval wrote comparison images**

Run: `ls logs/$(ls -t logs | head -1)/pred_spectrum/3000 | head`
Expected: several `N.png` comparison images (pred vs GT) from `paint_spectrum_compare`.

- [ ] **Step 3 (gate):** If loss is NaN or flat, or SSIM does not exceed a constant-image baseline, STOP and debug (most likely target misalignment — re-check Task 7 — or `tx_pos` scale/cull — re-check Task 8 `min_tx_norm`). Do not proceed to a full run. No commit (this step produces logs only).

---

## Task 10: Full per-scene training + reporting (§9.5/§9.6)

- [ ] **Step 1: Run a full per-scene training**

Run:
```bash
cd /home/hanyan_arch/jaehyeon/WRF-GSplus
python train.py --datadir ./data_mw_town05_parkinglot \
  --iterations 200000 --disable_viewer \
  2>&1 | tee /tmp/mw_train_full.log
```
Expected: completes; periodic evaluations log median SSIM and mean pixel error to `logs/<timestamp>/logger.log` and write `all_ssim.txt`.

- [ ] **Step 2: Compare against the mean-spectrum baseline**

Run:
```bash
python - <<'PY'
import glob, numpy as np, imageio.v2 as imageio
from skimage.metrics import structural_similarity as ssim
files = sorted(glob.glob('data_mw_town05_parkinglot/spectrum/*.png'))
imgs = np.stack([imageio.imread(f).astype(np.float64)/255.0 for f in files])
mean_img = imgs.mean(0)
sims = [ssim(mean_img, im, data_range=1.0) for im in imgs]
print('mean-spectrum baseline SSIM: median={:.4f} mean={:.4f}'.format(np.median(sims), np.mean(sims)))
PY
```
Expected: prints the baseline. **Success criterion:** the trained model's median SSIM (from `logger.log`) must exceed this baseline — confirming the model learned genuine `tx_pos`-conditioned structure, not just the average spectrum.

- [ ] **Step 3 (optional reporting):** Record final median SSIM, mean pixel error, and the baseline in a short note under `logs/` or the PR description. No code commit required.

---

## Task 11: Final verification sweep

- [ ] **Step 1: Run the whole unit suite**

Run: `cd /home/hanyan_arch/jaehyeon/WRF-GSplus && python -m pytest tests/ -q`
Expected: all tests PASS (geometry 7, spectrum 6, dataio 5, assemble 4, arguments 2).

- [ ] **Step 2: Confirm the model/renderer were not structurally changed**

Run: `git diff --stat main -- gaussian_renderer scene/gaussian_model.py scene/deform_model.py utils/pos_utils.py submodules`
Expected: empty (no changes to the renderer, Gaussian model, deform network, or CUDA submodules — only `scene/__init__.py`, `arguments/__init__.py`, `train.py` were touched on the model side).

- [ ] **Step 3: Commit any final notes and open the PR**

```bash
git add -A && git commit -m "docs: record mw adaptation results" || true
```
Then create a PR from `wrfgs-multimodal-adaptation` summarizing: converter usage, the alignment+calibration gates, and the SSIM-vs-baseline result.

---

## Notes for the executor

- **Per-scene retraining:** to target another V2I `sunny` scenario, re-run Task 8 Step 1 with different `--town/--scenario` and a fresh `--out-dir`, then Task 10 with that `--datadir`. No code changes.
- **Gates are real:** Task 7 (alignment) and Task 9 (overfit) are stop-the-line checks. A misaligned target or a culled `tx_pos` produces silently wrong training.
- **Do not commit converted datasets or `logs/`** (build artifacts; `.gitignore` covers `data_mw_*`).
- **Coordinate-frame reminder (spec §2.4/§6):** `tx_pos` only conditions the DeformNetwork; the camera is a fixed identity gauge. Any consistent fixed transform of `tx_pos` is fine — consistency and the measured `scale` are what matter.
