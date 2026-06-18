"""Extract a clean STATIC scene point cloud from an RSU's lidar.

The WRF-GS+ channel depends only on the static scene geometry (we verified the
spectrum is invariant to the moving vehicles). The RSU lidar is ~93% static; the
moving cars are a small, inconsistent minority. This tool aggregates several
frames of the RSU lidar (already in the RSU-local frame, the same frame as the
spectrum) and keeps only points that are consistently present across frames
(static), then voxel-downsamples to a target count. The result is the scene
geometry used to initialise the per-scene Gaussians for the scene-conditioned
model.
"""
import argparse
import glob
import os

import numpy as np


def load_pcd_xyz(path):
    lines = open(path).read().splitlines()
    di = next(i for i, l in enumerate(lines) if l.startswith("DATA")) + 1
    return np.array([[float(v) for v in l.split()[:3]] for l in lines[di:] if l.strip()],
                    dtype=np.float64)


def static_scene(rsu_dir, n_frames=20, voxel=0.5, occ_frac=0.5, max_points=40000):
    """Voxel-occupancy static extraction.

    Aggregate n_frames lidar scans; a voxel is 'static' if it is occupied in at
    least occ_frac of the frames. Return the centroid of each static voxel.
    """
    files = sorted(glob.glob(os.path.join(rsu_dir, "*.pcd")))
    if not files:
        raise FileNotFoundError("no .pcd under " + rsu_dir)
    step = max(1, len(files) // n_frames)
    files = files[::step][:n_frames]

    counts = {}      # voxel key -> frames seen in
    sums = {}        # voxel key -> summed xyz
    for f in files:
        pts = load_pcd_xyz(f)
        keys = np.floor(pts / voxel).astype(np.int64)
        seen = set()
        for p, k in zip(pts, map(tuple, keys)):
            if k not in seen:               # count each voxel once per frame
                seen.add(k)
                counts[k] = counts.get(k, 0) + 1
            sums.setdefault(k, [np.zeros(3), 0])
            sums[k][0] += p
            sums[k][1] += 1

    thresh = occ_frac * len(files)
    pts = np.array([sums[k][0] / sums[k][1] for k, c in counts.items() if c >= thresh])
    if len(pts) > max_points:                # uniform subsample to the cap
        idx = np.random.RandomState(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts, len(files)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/NHNHOME/WORKSPACE/0526040099_A/jaehyeon/multimodal_wireless")
    ap.add_argument("--weather", default="sunny")
    ap.add_argument("--town", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--rsu", default="rsu_1")
    ap.add_argument("--out", required=True, help="output .npy of (N,3) static scene points")
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--max-points", type=int, default=40000)
    args = ap.parse_args(argv)

    # Sensor Data scenario dir may add a descriptor before the seed suffix
    pat = os.path.join(args.root, args.weather, "Sensor Data", args.town, args.scenario + "*_seed*")
    matches = sorted(glob.glob(pat))
    assert len(matches) == 1, "expected one sensor dir, got {}".format(matches)
    rsu_dir = os.path.join(matches[0], args.rsu)

    pts, nf = static_scene(rsu_dir, args.n_frames, args.voxel, max_points=args.max_points)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, pts.astype(np.float32))
    r = np.linalg.norm(pts, axis=1)
    print("[extract] {}/{}: {} static points from {} frames (voxel {}m)".format(
        args.town, args.scenario, len(pts), nf, args.voxel))
    print("[extract]   range {:.1f}-{:.0f} m, z[{:.1f},{:.1f}] -> {}".format(
        r.min(), r.max(), pts[:, 2].min(), pts[:, 2].max(), args.out))


if __name__ == "__main__":
    main()
