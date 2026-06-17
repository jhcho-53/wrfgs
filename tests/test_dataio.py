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
