"""Qualitative paper figure: zero-shot held-out spectra (unseen scene) for
GT vs absolute-coord model vs relative-geometry model. Trains both small shared
MLPs on 2 scenes and renders the held-out 3rd. Run inside wrfgsplus on a GPU."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams
from train_multiscene import SceneHolder, predict          # relative predict
from scene.deform_model import DeformModel
from utils.rel_deform import RelDeformModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim

parser = ArgumentParser()
lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
args = parser.parse_args([])
opt = op.extract(args); pipe = pp.extract(args)
bg = torch.zeros(3, device="cuda")
P = "scenes_static"

train = [SceneHolder("data_mw_town05_parkinglot", f"{P}/Town05_parkinglot.npy"),
         SceneHolder("data_mw_Town05_ringroad", f"{P}/Town05_ringroad.npy")]
hold = SceneHolder("data_mw_Town05_CBDcrossroad", f"{P}/Town05_CBDcrossroad.npy")


def predict_abs(sc, deform, tx_pos):
    N = sc.gaussians.get_xyz.shape[0]
    ti = tx_pos.cuda().reshape(1, -1).expand(N, -1)
    d = deform.step(sc.gaussians.get_xyz.detach(), ti)
    img = render(sc.cam, sc.gaussians, pipe, bg, *d, use_trained_exp=False, separate_sh=False)["render"]
    return torch.abs(img[0] + 1j * img[1])


def loss_of(pred, gt):
    return (1 - opt.lambda_dssim) * l1_loss(pred, gt) + opt.lambda_dssim * (1 - fused_ssim(pred[None, None], gt[None, None]))


rel = RelDeformModel(); rel.train_setting(opt)
abs_ = DeformModel(); abs_.train_setting(opt)
N_IT = 2500
for it in range(1, N_IT + 1):
    sc = train[it % 2]
    spectrum, tx = sc.next(); gt = spectrum.cuda().squeeze()
    lr = loss_of(predict(sc, rel, pipe, bg, tx), gt)
    lr.backward(); rel.optimizer.step(); rel.optimizer.zero_grad(); rel.update_learning_rate(it)
    la = loss_of(predict_abs(sc, abs_, tx), gt)
    la.backward(); abs_.optimizer.step(); abs_.optimizer.zero_grad(); abs_.update_learning_rate(it)
    if it % 500 == 0:
        print(f"  trained {it}/{N_IT}", flush=True)

rel.deform.eval(); abs_.deform.eval()
rows = []
loader = iter(hold.test_loader)
for k in range(4):
    spectrum, tx = next(loader); gt = spectrum.cuda().squeeze()
    with torch.no_grad():
        pr = predict(hold, rel, pipe, bg, tx)
        pa = predict_abs(hold, abs_, tx)
    rows.append((gt.cpu().numpy(), pa.cpu().numpy(), pr.cpu().numpy()))

titles = ["Ground truth", "Absolute (memorizes)", "Relative geometry (ours)"]
fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.3 * len(rows)))
for i, (gt, pa, pr) in enumerate(rows):
    for j, im in enumerate([gt, pa, pr]):
        ax = axes[i, j]
        ax.imshow(im, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(titles[j], fontsize=12)
    axes[i, 0].set_ylabel(f"CBD #{i+1}\n(az↔, el↕)", fontsize=9)
fig.suptitle("Zero-shot held-out spectra on an UNSEEN scene (CBDcrossroad): relative matches GT, absolute collapses",
             fontsize=11.5, y=1.005)
fig.tight_layout()
fig.savefig("docs/research/figures/fig4_qualitative.png", bbox_inches="tight", dpi=170)
fig.savefig("docs/research/figures/fig4_qualitative.pdf", bbox_inches="tight")
print("saved fig4_qualitative")
