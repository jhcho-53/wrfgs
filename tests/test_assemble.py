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
