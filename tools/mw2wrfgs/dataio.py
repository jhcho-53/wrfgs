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
    # some Sensor Data dirs add a descriptor between scenario and seed
    # (e.g. Town03_crossroad -> Town03_crossroad_wiz_slope_seed42)
    pattern = os.path.join(root, weather, "Sensor Data", town, scenario + "*_seed*")
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
    scns = cfg["scenarios"]
    # config keys may carry a descriptor (Town03_crossroad -> Town03_crossroad_wiz_slope)
    if scenario in scns:
        key = scenario
    else:
        cands = [k for k in scns if k.startswith(scenario)]
        key = cands[0] if cands else next(iter(scns))
    t = scns[key]["rsu_transform"]
    loc = t["location"]
    yaw = float(t["rotation"]["yaw"])
    return np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64), yaw


def save_spectrum_png(path, uint8_img):
    """Write a (90,360) uint8 grayscale PNG (mode 'L'), matching data_test200."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, uint8_img)
