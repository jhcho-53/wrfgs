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
    """Max ||cav - rsu|| over all samples (the single fixed tx_pos scale, spec 6)."""
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
