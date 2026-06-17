import numpy as np

IMG_H = 90
IMG_W = 360


def per_path_power(a):
    """Intrinsic per-path power from Sionna gains (spec 5.1).

    a: complex array (1, 1, Nr, 1, Nt, n_paths, 1). Antenna dims carry only
    phase, so |a| is constant across them; element [0,0,0,0,0,k,0] suffices.
    Returns a real (n_paths,) array; zero-|a| (padded) slots come out 0.
    """
    a0 = a[0, 0, 0, 0, 0, :, 0]
    return np.abs(a0).astype(np.float64) ** 2


def synthesize_spectrum(p, row, col, sigma_az=8.0, sigma_el=8.0, floor_abs=0.03):
    """Densified 2D DoA spectrum (spec 5.3/5.4).

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
