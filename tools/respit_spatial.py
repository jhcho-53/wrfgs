"""Re-split an existing data_test200-format dataset by a SPATIAL holdout.

The WRF-GS+ channel is a (near-)deterministic function of the transmitter
position, so a random frame split leaks: every test frame has a near-identical
train frame one step away on the trajectory (nearest-neighbour SSIM ~0.9998).
This tool rewrites train_index.txt / test_index.txt using a geographic holdout:
project tx_pos onto its principal spatial axis, hold out one contiguous end as
test, and drop a buffer band around the cut so no test sample sits within
`buffer` of any train sample. That forces genuine spatial extrapolation.

PCA is rotation/scale invariant, so operating on the normalised tx_pos.csv is
equivalent to operating on world positions.
"""
import argparse
import os

import numpy as np
import pandas as pd


def spatial_split(xy, test_frac=0.2, buffer=6.0, scale=1.0):
    """Return (train_mask, test_mask, drop_mask) for a principal-axis end holdout.

    xy: (N, 2) planar positions. `buffer` and `scale` are in the same units as
    xy after multiplying by `scale` (use scale to convert normalised tx_pos back
    to metres for an interpretable buffer).
    """
    p = np.asarray(xy, dtype=np.float64) * scale
    pc = p - p.mean(0)
    # principal axis via SVD
    _, _, vt = np.linalg.svd(pc, full_matrices=False)
    t = pc @ vt[0]
    cut = np.percentile(t, 100 * (1.0 - test_frac))
    test = t >= cut + buffer / 2.0
    train = t < cut - buffer / 2.0
    drop = ~(test | train)
    return train, test, drop


def main(argv=None):
    ap = argparse.ArgumentParser(description="Spatial-holdout re-split of a WRF-GS+ dataset")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--buffer", type=float, default=6.0, help="separation band in metres")
    ap.add_argument("--scale", type=float, default=75.486,
                    help="metres per normalised tx_pos unit (the converter's scale)")
    ap.add_argument("--write", action="store_true", help="overwrite train/test index files")
    args = ap.parse_args(argv)

    tx = pd.read_csv(os.path.join(args.datadir, "tx_pos.csv")).values
    ids = ["{:05d}".format(i + 1) for i in range(len(tx))]  # 1-based, row i -> id i+1
    train, test, drop = spatial_split(tx[:, :2], args.test_frac, args.buffer, args.scale)

    # leakage / separation check
    from scipy.spatial import cKDTree
    p = tx[:, :2] * args.scale
    tree = cKDTree(p[train])
    dmin, _ = tree.query(p[test])
    print("[respit] train={} test={} drop={} (test_frac={}, buffer={}m)".format(
        train.sum(), test.sum(), drop.sum(), args.test_frac, args.buffer))
    print("[respit] min test->train distance = {:.2f} m (>= buffer means no leakage)".format(dmin.min()))

    if args.write:
        tr = sorted(np.array(ids)[train].tolist())
        te = sorted(np.array(ids)[test].tolist())
        with open(os.path.join(args.datadir, "train_index.txt"), "w") as f:
            f.write("\n".join(tr) + "\n")
        with open(os.path.join(args.datadir, "test_index.txt"), "w") as f:
            f.write("\n".join(te) + "\n")
        print("[respit] wrote train_index.txt ({}) and test_index.txt ({})".format(len(tr), len(te)))
    else:
        print("[respit] dry run (pass --write to overwrite index files)")


if __name__ == "__main__":
    main()
