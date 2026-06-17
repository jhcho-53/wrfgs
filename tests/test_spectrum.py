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
