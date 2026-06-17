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
