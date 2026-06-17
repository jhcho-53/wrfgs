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
    """RSU-local, scaled conditioning position (spec section 6).

    tx = R_rsu^{-1} . (cav - rsu) / scale, R_rsu = yaw rotation about world z.
    Returns a (3,) float64 array.
    """
    cav = np.asarray(cav_xyz, dtype=np.float64)
    rsu = np.asarray(rsu_xyz, dtype=np.float64)
    rel = cav - rsu
    r_inv = Rotation.from_euler("z", rsu_yaw_deg, degrees=True).inv()
    return r_inv.apply(rel) / float(scale)
