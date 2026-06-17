"""Compare synthesized vs reference (data_test200) brightness stats (spec 5.5).
Reports nonzero-fraction, mean-norm, fraction>0.5 over N images and whether they
fall in the acceptance bands. Use it to tune --sigma-az/--sigma-el/--floor-abs."""
import argparse
import glob
import os

import numpy as np
import imageio.v2 as imageio


def stats(folder, n=50):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))[:n]
    nz, mn, hi = [], [], []
    for f in files:
        im = imageio.imread(f).astype(np.float64) / 255.0
        nz.append((im > 0).mean())
        mn.append(im.mean())
        hi.append((im > 0.5).mean())
    return np.mean(nz), np.mean(mn), np.mean(hi), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="converted spectrum/ dir")
    ap.add_argument("--ref", default="./data_test200/spectrum")
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    s = stats(args.synth, args.n)
    r = stats(args.ref, args.n)
    print("synth nonzero={:.3f} mean={:.3f} frac>0.5={:.3f} (n={})".format(*s))
    print("ref   nonzero={:.3f} mean={:.3f} frac>0.5={:.3f} (n={})".format(*r))
    bands = (s[0] >= 0.98, 0.30 <= s[1] <= 0.55, 0.20 <= s[2] <= 0.45)
    print("acceptance (nonzero>=.98, mean in[.30,.55], frac>.5 in[.20,.45]):", bands)
    print("CALIBRATION", "PASS" if all(bands) else "ADJUST sigma/floor")


if __name__ == "__main__":
    main()
