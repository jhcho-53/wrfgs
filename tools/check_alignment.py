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
