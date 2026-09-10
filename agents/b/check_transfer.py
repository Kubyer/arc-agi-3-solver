"""Transfer check for the learned success signal (round 3).

Loads success_prior.json and a directory of SUCCESS_DUMP npz files, then
asks: does the toy-trained V rank real-game states sensibly?
Per dump file:
  - mean V on late-winning states (y > 0.5) vs on failure terminals (y==0)
  - corr(V, y) restricted to winning states (y > 0): does V rise along
    winning trajectories?

Usage: .venv/bin/python agents/b/check_transfer.py <dumps_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from b.success import SuccessModel  # noqa: E402


def main():
    dumpdir = sys.argv[1]
    m = SuccessModel.from_prior()
    print("prior: %s" % m.prior_info)
    print("top features: %s" % (m.top_features(8),))
    for fn in sorted(os.listdir(dumpdir)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(dumpdir, fn))
        X, y = z["X"], z["y"]
        if len(X) == 0:
            continue
        V = np.array([m.value(x) for x in X])
        win = y > 0
        fail = y == 0
        late = y > 0.5
        c = float(np.corrcoef(V[win], y[win])[0, 1]) if win.sum() > 3 else float("nan")
        print("%-18s n=%4d  V|late-win=%.3f  V|fail=%.3f  corr(V,y|win)=%+.3f" %
              (fn, len(X), V[late].mean() if late.any() else float("nan"),
               V[fail].mean() if fail.any() else float("nan"), c))


if __name__ == "__main__":
    main()
