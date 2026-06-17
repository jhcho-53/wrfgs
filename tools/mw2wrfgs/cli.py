import argparse
import os

import numpy as np

from . import dataio, assemble


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Multimodal-Wireless V2I -> WRF-GS+ dataset")
    ap.add_argument("--root", default="/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless")
    ap.add_argument("--weather", default="sunny")
    ap.add_argument("--antenna-config", default="Nt_1_64_Nr_1_16_fc_28GHz")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--scenario", default="Town05_parkinglot")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sensor-dir", default=None, help="override seed-glob resolution")
    ap.add_argument("--sigma-az", type=float, default=8.0)
    ap.add_argument("--sigma-el", type=float, default=8.0)
    ap.add_argument("--floor-abs", type=float, default=0.03)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tx-norm", type=float, default=0.2,
                    help="assert min ||tx_pos|| exceeds this (cull-radius margin, spec 6)")
    args = ap.parse_args(argv)

    ch = dataio.channel_scenario_dir(args.root, args.weather, args.antenna_config,
                                     args.town, args.scenario)
    sen = args.sensor_dir or dataio.resolve_sensor_dir(
        args.root, args.weather, args.town, args.scenario)
    config_yaml = os.path.join(sen, "config.yaml")
    rsu_xyz, rsu_yaw = dataio.load_rsu_pose(config_yaml, args.scenario)

    samples = assemble.enumerate_samples(ch, sen)
    if not samples:
        raise SystemExit("no (cav, frame) samples found under {}".format(ch))
    scale = assemble.measure_scale(samples, rsu_xyz)
    print("[mw2wrfgs] samples={}  rsu={}  yaw={}  scale={:.3f}".format(
        len(samples), rsu_xyz.tolist(), rsu_yaw, scale))

    stats = assemble.write_dataset(
        args.out_dir, samples, rsu_xyz, rsu_yaw, scale,
        args.sigma_az, args.sigma_el, args.floor_abs,
        train_ratio=args.train_ratio, seed=args.seed,
        dataset_name="{}_{}".format(args.town, args.scenario),
    )
    print("[mw2wrfgs] wrote {} -> {}".format(stats, args.out_dir))

    if stats["min_tx_norm"] <= args.min_tx_norm:
        raise SystemExit(
            "min ||tx_pos|| = {:.4f} <= {} : too close to cull radius; "
            "inspect scenario geometry".format(stats["min_tx_norm"], args.min_tx_norm))


if __name__ == "__main__":
    main()
