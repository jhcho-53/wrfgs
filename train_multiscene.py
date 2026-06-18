"""Scene-conditioned WRF-GS+ training (cross-scene).

Instead of 200k random Gaussians memorising ONE scene, each scene's Gaussians
are FROZEN on its static RSU-lidar geometry (the per-scene input) and a SINGLE
shared DeformModel maps (point xyz, tx_pos) -> per-Gaussian EM signal/shape.
Trained jointly over several scenes; a held-out scene just plugs in its lidar.

A-strict variant: only the shared DeformModel is learnt (no per-scene params,
d_xyz not applied by the renderer). Run the capacity check on a few scenes; if
it cannot fit them, enable d_xyz (A+deform) so the field can place energy at the
tx-dependent LoS direction.

Each --scenes entry is "DATADIR:LIDAR_NPY". DATADIR is a converted dataset
(spectrum/, tx_pos.csv, gateway_info.yml identity gauge); LIDAR_NPY is the
static RSU scene cloud from tools/extract_rsu_scene.py.
"""
import os
import numpy as np
import torch
import yaml
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataloader import dataset_dict
from gaussian_renderer import render
from utils.generate_camera import generate_new_cam
from utils.loss_utils import l1_loss, ssim
from utils.rel_deform import RelDeformModel
import json, sys


def scene_scale_yaw(datadir, scenario):
    """Per-scene tx-normalisation scale + RSU yaw (cached in datadir/meta.json),
    needed to map normalised tx_pos back to the Rx-frame metres of the lidar."""
    meta_p = os.path.join(datadir, "meta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p)); return m["scale"], m["yaw"]
    sys.path.insert(0, "tools")
    from mw2wrfgs import dataio
    from mw2wrfgs.assemble import enumerate_samples, measure_scale
    root = "/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless"
    town = scenario.split("_")[0]
    ch = dataio.channel_scenario_dir(root, "sunny", "Nt_1_64_Nr_1_16_fc_28GHz", town, scenario)
    sen = dataio.resolve_sensor_dir(root, "sunny", town, scenario)
    rsu, yaw = dataio.load_rsu_pose(os.path.join(sen, "config.yaml"), scenario)
    scale = measure_scale(enumerate_samples(ch, sen), rsu)
    json.dump({"scale": scale, "yaw": yaw}, open(meta_p, "w"))
    return scale, yaw

try:
    from fused_ssim import fused_ssim
    FUSED = True
except Exception:
    FUSED = False


def to_cam_frame(pts):
    """CARLA/world z-up -> renderer camera y-up: cam = [x, z, y]. Same axis
    permutation P the GT synthesis used (angles_to_pixel), so lidar Gaussians
    and the target spectrum live in one consistent frame."""
    return np.asarray(pts)[:, [0, 2, 1]]


class SceneHolder:
    def __init__(self, datadir, lidar_npy, sh_degree=0):
        with open(os.path.join(datadir, "gateway_info.yml")) as f:
            gw = yaml.safe_load(f)["gateway1"]
        R = torch.from_numpy(Rotation.from_quat(gw["orientation"]).as_matrix()).float()
        self.cam = generate_new_cam(R, gw["position"])          # fixed per-scene camera
        ds = dataset_dict["rfid"]
        self.train_set = ds(datadir, os.path.join(datadir, "train_index.txt"))
        self.test_set = ds(datadir, os.path.join(datadir, "test_index.txt"))
        self.train_loader = DataLoader(self.train_set, batch_size=1, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=0)
        self._it = iter(self.train_loader)
        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.init_from_lidar(to_cam_frame(np.load(lidar_npy)))
        self.name = os.path.basename(datadir.rstrip("/"))
        # tx_pos (normalised, R_z(yaw)^-1 frame) -> Rx-frame metres, for relative geometry
        scenario = os.path.basename(lidar_npy).replace(".npy", "")
        self.scale, yaw = scene_scale_yaw(datadir, scenario)
        self.Rz = torch.tensor(Rotation.from_euler("z", yaw, degrees=True).as_matrix(),
                               dtype=torch.float32, device="cuda")

    def tx_to_cam(self, tx_pos):
        w = self.Rz @ (tx_pos.cuda().reshape(3).float() * self.scale)  # world (cav-rsu)
        return w[[0, 2, 1]]                                            # camera frame (y-up)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.train_loader)
            return next(self._it)


def predict(sc, deform, pipe, bg, tx_pos):
    tx_cam = sc.tx_to_cam(tx_pos)                                 # Tx in Rx-frame metres
    d_xyz, d_rot, d_scl, d_sig = deform.step(sc.gaussians.get_xyz, tx_cam)
    img = render(sc.cam, sc.gaussians, pipe, bg, d_xyz, d_rot, d_scl, d_sig,
                 use_trained_exp=False, separate_sh=False)["render"]
    return torch.abs(img[0] + 1j * img[1])                       # (90, 360) magnitude


@torch.no_grad()
def evaluate(scenes, deform, pipe, bg, n_max=120):
    deform.deform.eval()
    out = {}
    for sc in scenes:
        sims = []
        for k, (spectrum, tx_pos) in enumerate(sc.test_loader):
            if k >= n_max:
                break
            pred = predict(sc, deform, pipe, bg, tx_pos)
            gt = spectrum.cuda().squeeze()
            sims.append(float(fused_ssim(pred[None, None], gt[None, None]) if FUSED else ssim(pred, gt)))
        out[sc.name] = float(np.median(sims))
    deform.deform.train()
    return out


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--scenes", nargs="+", required=True, help="train DATADIR:LIDAR_NPY ...")
    parser.add_argument("--holdout-scenes", nargs="+", default=[],
                        help="eval-only DATADIR:LIDAR_NPY (zero-shot, never trained)")
    parser.add_argument("--eval-every", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1, help="render graphs per step (VRAM x B)")
    args = parser.parse_args()
    opt = op.extract(args); pipe = pp.extract(args)
    n_iter = opt.iterations  # from the standard --iterations (OptimizationParams)

    scenes = [SceneHolder(*s.split(":")) for s in args.scenes]
    holdout = [SceneHolder(*s.split(":")) for s in args.holdout_scenes]
    print("[multiscene-REL] train scenes: {} | HOLD-OUT (zero-shot): {}".format(
        [s.name for s in scenes], [s.name for s in holdout]))
    deform = RelDeformModel()
    deform.train_setting(opt)
    bg = torch.zeros(3, device="cuda")

    ema = 0.0
    for it in range(1, n_iter + 1):
        # batch: keep B render graphs alive (uses ~B x render VRAM) for a
        # larger effective batch / better gradients; scenes mixed within a step
        loss = 0.0
        for b in range(args.batch):
            sc = scenes[(it * args.batch + b) % len(scenes)]
            spectrum, tx_pos = sc.next()
            pred = predict(sc, deform, pipe, bg, tx_pos)
            gt = spectrum.cuda().squeeze()
            l1 = l1_loss(pred, gt)
            sv = fused_ssim(pred[None, None], gt[None, None]) if FUSED else ssim(pred, gt)
            loss = loss + (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - sv)
        loss = loss / args.batch
        loss.backward()
        deform.optimizer.step()
        deform.optimizer.zero_grad()
        deform.update_learning_rate(it)

        ema = 0.4 * loss.item() + 0.6 * ema
        if it % 200 == 0:
            print("[it {:6d}] loss(ema)={:.5f}".format(it, ema), flush=True)
        if it % args.eval_every == 0 or it == n_iter:
            res = evaluate(scenes, deform, pipe, bg)
            print("[it {:6d}] TRAIN-scene median SSIM: {}".format(
                it, {k: round(v, 4) for k, v in res.items()}), flush=True)
            if holdout:
                hz = evaluate(holdout, deform, pipe, bg)
                print("[it {:6d}] *** ZERO-SHOT held-out median SSIM: {} ***".format(
                    it, {k: round(v, 4) for k, v in hz.items()}), flush=True)


if __name__ == "__main__":
    main()
