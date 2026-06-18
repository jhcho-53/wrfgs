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
import random
import time
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


def angular_err(pred, gt):
    """Power-weighted DoA error (degrees): does the predicted energy sit at the
    right azimuth/elevation? Discriminative even when SSIM saturates on smooth
    targets. pred/gt: (90,360) magnitude images on cuda."""
    import math
    H, W = 90, 360
    dev = pred.device
    az = (torch.arange(W, device=dev, dtype=torch.float32) / W * 2 - 1) * math.pi
    el = (torch.arange(H, device=dev, dtype=torch.float32) / H * 2 - 1) * (math.pi / 2)

    def caz(img):
        w = img.sum(0)
        return torch.atan2((w * torch.sin(az)).sum(), (w * torch.cos(az)).sum())

    def cel(img):
        w = img.sum(1)
        return (w * el).sum() / (w.sum() + 1e-6)
    d = caz(pred) - caz(gt)
    d = torch.abs((d + math.pi) % (2 * math.pi) - math.pi)
    return float(d * 180 / math.pi), float(torch.abs(cel(pred) - cel(gt)) * 180 / math.pi)


@torch.no_grad()
def evaluate(scenes, deform, pipe, bg, n_max=120):
    deform.deform.eval()
    out = {}
    for sc in scenes:
        sims, aze, ele = [], [], []
        for k, (spectrum, tx_pos) in enumerate(sc.test_loader):
            if k >= n_max:
                break
            pred = predict(sc, deform, pipe, bg, tx_pos)
            gt = spectrum.cuda().squeeze()
            sims.append(float(fused_ssim(pred[None, None], gt[None, None]) if FUSED else ssim(pred, gt)))
            a, e = angular_err(pred, gt)
            aze.append(a); ele.append(e)
        out[sc.name] = {"ssim": round(float(np.median(sims)), 4),
                        "az_err_deg": round(float(np.median(aze)), 1),
                        "el_err_deg": round(float(np.median(ele)), 1)}
    deform.deform.train()
    return out


def set_seed(seed):
    """Make the shared-MLP init and the per-scene DataLoader shuffling
    deterministic so a re-run reproduces the reported DoA numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--scenes", nargs="+", required=True, help="train DATADIR:LIDAR_NPY ...")
    parser.add_argument("--holdout-scenes", nargs="+", default=[],
                        help="eval-only DATADIR:LIDAR_NPY (zero-shot, never trained)")
    parser.add_argument("--eval-every", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1, help="render graphs per step (VRAM x B)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--out-dir", default=None,
                        help="dir for checkpoints + results.json (default runs_multiscene/<ts>)")
    parser.add_argument("--load-checkpoint", default=None,
                        help="path to a deform_*.pth to load before train/eval")
    parser.add_argument("--eval-only", action="store_true",
                        help="load --load-checkpoint, evaluate train+holdout, write results, exit")
    args = parser.parse_args()
    opt = op.extract(args); pipe = pp.extract(args)
    n_iter = opt.iterations  # from the standard --iterations (OptimizationParams)

    set_seed(args.seed)
    out_dir = args.out_dir or os.path.join("runs_multiscene", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    scenes = [SceneHolder(*s.split(":")) for s in args.scenes]
    holdout = [SceneHolder(*s.split(":")) for s in args.holdout_scenes]
    print("[multiscene-REL] train scenes: {} | HOLD-OUT (zero-shot): {}".format(
        [s.name for s in scenes], [s.name for s in holdout]))
    print("[multiscene-REL] out_dir={} seed={} iters={}".format(out_dir, args.seed, n_iter), flush=True)
    deform = RelDeformModel()
    deform.train_setting(opt)
    bg = torch.zeros(3, device="cuda")

    # config manifest so the run is self-describing on disk
    config = {
        "scenes": args.scenes, "holdout_scenes": args.holdout_scenes,
        "iterations": n_iter, "seed": args.seed, "eval_every": args.eval_every,
        "batch": args.batch, "lambda_dssim": opt.lambda_dssim,
        "position_lr_init": opt.position_lr_init, "position_lr_final": opt.position_lr_final,
        "spatial_lr_scale": deform.spatial_lr_scale,
    }
    history = []  # [{iteration, train, holdout}] across eval checkpoints

    def dump_results():
        json.dump({"config": config, "history": history},
                  open(os.path.join(out_dir, "results.json"), "w"), indent=2)

    def do_eval(it):
        res = evaluate(scenes, deform, pipe, bg)
        print("[it {:6d}] TRAIN-scene: {}".format(it, res), flush=True)
        hz = evaluate(holdout, deform, pipe, bg) if holdout else {}
        if holdout:
            print("[it {:6d}] *** ZERO-SHOT held-out: {} ***".format(it, hz), flush=True)
        history.append({"iteration": it, "train": res, "holdout": hz})
        dump_results()

    if args.load_checkpoint:
        deform.load(args.load_checkpoint)
        print("[multiscene-REL] loaded checkpoint {}".format(args.load_checkpoint), flush=True)
    if args.eval_only:
        do_eval(0)
        print("[multiscene-REL] eval-only done -> {}/results.json".format(out_dir), flush=True)
        return

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
            do_eval(it)
            deform.save(os.path.join(out_dir, "deform_{}.pth".format(it)))
            deform.save(os.path.join(out_dir, "deform_latest.pth"))

    print("[multiscene-REL] done -> {} (deform_latest.pth, results.json)".format(out_dir), flush=True)


if __name__ == "__main__":
    main()
