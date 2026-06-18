"""Relative-geometry signal field for the SCENE-CONDITIONED / cross-scene model.

The absolute-coordinate conditioning of the original DeformNetwork (point xyz +
tx_pos) memorises each scene and fails to generalise (consistent with RayProNet,
arXiv:2406.16907, which is per-scene w.r.t. geometry). GeNeRT (arXiv:2506.18295)
shows that RELATIVE geometric features (incident/outgoing directions, angles)
enable inter-scenario zero-shot generalisation. This module reconditions the
per-Gaussian signal MLP on the single-bounce relative geometry of each scatterer
(a lidar point) w.r.t. the Rx (origin) and the Tx, which is scene-agnostic, so a
held-out scene only supplies its lidar geometry.

Frame: points p and tx are both in the renderer CAMERA frame (Rx at the origin,
y = up, metres). d_rx = p/|p| is exactly the arrival direction the rasterizer
will project, so the MLP predicts "how much energy scatterer p sends to the Rx
for this Tx", and the renderer places it at the correct azimuth/elevation.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pos_utils import get_embedder
from utils.general_utils import get_expon_lr_func


def rel_geom_features(p, tx):
    """Per-scatterer single-bounce relative features. p:(N,3) Rx-frame metres,
    tx:(3,) same frame. Returns (N, 13): [d_rx(3), d_tx(3), bisector(3), cos_bounce,
    log1p r_rx, log1p r_tx, log1p path]."""
    eps = 1e-6
    r_rx = p.norm(dim=1, keepdim=True)
    d_rx = p / (r_rx + eps)                       # Rx -> scatterer = arrival direction
    v = tx.view(1, 3) - p
    r_tx = v.norm(dim=1, keepdim=True)
    d_tx = v / (r_tx + eps)                       # scatterer -> Tx direction
    cos_b = (d_rx * d_tx).sum(1, keepdim=True)    # bounce-angle cosine
    bis = d_rx + d_tx
    bis = bis / (bis.norm(dim=1, keepdim=True) + eps)   # specular-normal proxy
    return torch.cat([d_rx, d_tx, bis, cos_b,
                      torch.log1p(r_rx), torch.log1p(r_tx),
                      torch.log1p(r_rx + r_tx)], dim=1)


class RelDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, feat_dim=13, multires=6):
        super().__init__()
        self.embed, ch = get_embedder(multires, feat_dim)
        self.D = D
        self.skips = [D // 2]
        self.linear = nn.ModuleList(
            [nn.Linear(ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + ch, W)
                for i in range(D - 1)])
        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_signal = nn.Linear(W, 3)
        self.gaussian_phase = nn.Linear(W, 3)

    def forward(self, p, tx):
        h0 = self.embed(rel_geom_features(p, tx))
        h = h0
        for i, layer in enumerate(self.linear):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([h0, h], -1)
        d_xyz = self.gaussian_warp(h)
        rotation = self.gaussian_rotation(h)
        scaling = self.gaussian_scaling(h)
        signal = torch.abs(self.gaussian_signal(h) * torch.exp(1j * self.gaussian_phase(h)))
        return d_xyz, rotation, scaling, signal


class RelDeformModel:
    """Drop-in replacement for DeformModel but conditioned on relative geometry.
    step(p, tx): p = Gaussian positions (Rx-frame), tx = Tx position (Rx-frame)."""
    def __init__(self):
        self.deform = RelDeformNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, p, tx):
        return self.deform(p, tx)

    def train_setting(self, training_args):
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.deform.parameters()),
              "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "deform"}],
            lr=0.0, eps=1e-15)
        self.scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        for g in self.optimizer.param_groups:
            if g["name"] == "deform":
                g["lr"] = self.scheduler(iteration)
                return g["lr"]

    def save(self, path):
        """Persist the shared signal MLP weights (the only learnable state of the
        scene-conditioned model) so a run is reproducible without retraining."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.deform.state_dict(), path)

    def load(self, path):
        self.deform.load_state_dict(torch.load(path, map_location="cuda"))
