"""Offline prior training for the learned success signal (round 3).

Phase 1: run the toys (bt11, bt33) over seeds 0-2 with the learned signal
  disabled (SUCCESS_OFF=1) but feature logging+dump on (SUCCESS_DUMP) ->
  winning trajectories with dense progress labels y=i/(n-1).
Phase 2: run the dev-subset real games (ar25, g50t, lp85, lf52, vc33) at
  seed 0 the same way -> game-over terminals (y=0) plus any transient
  completions. The failures are the abundant negative class the sparse
  completion signal lacks.
Phase 3: fit L2 ridge regression (numpy closed form), save
  agents/b/success_prior.json, print diagnostics:
  - in-sample R^2, top features by |weight| (paper material),
  - cross-game transfer: train on bt11 only, R^2 on bt33.

Usage: .venv/bin/python agents/b/train_prior.py [outdir]
Dumps go to <outdir>/dumps (kept for the transfer check); the prior is
always written to agents/b/success_prior.json.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from b.benchmark import run_game, TOY_DIR  # noqa: E402
from b.success import (FEATURE_NAMES, N_FEATURES, _ridge_fit,  # noqa: E402
                       save_prior)
from arc_agi import Arcade, OperationMode  # noqa: E402

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REAL_DIR = os.path.join(CODE, "environment_files")
TOYS = ["bt11", "bt33"]
DEV = ["ar25", "g50t", "lp85", "lf52", "vc33"]


def run_dump(client, gid, env_dir, seed, dumpdir, max_actions=4000,
             max_wall_s=180):
    os.environ["SUCCESS_OFF"] = "1"
    os.environ["SUCCESS_DUMP"] = dumpdir
    os.environ["SUCCESS_DUMP_TAG"] = "%s_seed%d" % (gid, seed)
    t0 = time.time()
    res = run_game(client, gid, env_dir, max_actions=max_actions,
                   max_wall_s=max_wall_s, seed=seed)
    print("  %-6s seed=%d lvmax=%d act=%d wall=%ds stop=%s" %
          (gid, seed, res.get("levels_ever", 0), res["actions"],
           int(time.time() - t0), res["stop"]),
          flush=True)
    return res


def load_dumps(dumpdir):
    Xs, ys, tags = [], [], []
    for fn in sorted(os.listdir(dumpdir)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(dumpdir, fn))
        X, y = z["X"], z["y"]
        if len(X) == 0:
            continue
        assert X.shape[1] == N_FEATURES, (fn, X.shape)
        Xs.append(X)
        ys.append(y)
        tags.extend([fn] * len(X))
    if not Xs:
        return None, None, []
    return np.vstack(Xs), np.concatenate(ys), tags


def r2(y, yhat):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss if ss > 0 else 0.0


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hidden_files",
        "prior_train")
    dumpdir = os.path.join(outdir, "dumps")
    os.makedirs(dumpdir, exist_ok=True)
    print("dumps -> %s" % dumpdir)

    toy_client = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=TOY_DIR)
    real_client = Arcade(operation_mode=OperationMode.OFFLINE,
                         environments_dir=REAL_DIR)

    print("phase 1: toys (winning trajectories)")
    for gid in TOYS:
        for seed in (0, 1, 2):
            run_dump(toy_client, gid, TOY_DIR, seed, dumpdir)
    print("phase 2: dev real games (failure terminals + rare completions)")
    for gid in DEV:
        run_dump(real_client, gid, REAL_DIR, 0, dumpdir)

    X, y, tags = load_dumps(dumpdir)
    print("total samples: %d (pos y>0.05: %d, y==0: %d)" %
          (len(X), int((y > 0.05).sum()), int((y == 0).sum())))
    npos = int((y > 0.05).sum())
    if npos < 20:
        print("ABORT: too few positive samples to train a prior")
        sys.exit(2)

    # class balance: failures are abundant; cap y==0 at 3x positives so
    # the regression doesn't collapse to predicting 0.
    rng = np.random.RandomState(0)
    zero_i = np.where(y == 0)[0]
    pos_i = np.where(y > 0)[0]
    keep_zero = rng.choice(zero_i, size=min(len(zero_i), 3 * len(pos_i)),
                           replace=False)
    keep = np.sort(np.concatenate([pos_i, keep_zero]))
    Xb, yb = X[keep], y[keep]
    print("balanced: %d samples (%d pos)" % (len(Xb), len(pos_i)))

    w = _ridge_fit(Xb, yb, np.zeros(N_FEATURES), strength=1.0)
    print("in-sample R^2: %.3f" % r2(yb, Xb @ w))
    order = np.argsort(-np.abs(w))
    print("top features by |weight|:")
    for i in order[:10]:
        print("  %-22s %+.4f" % (FEATURE_NAMES[i], w[i]))

    # cross-game transfer: train on bt11 only, test on bt33
    b11 = np.array([t.startswith("bt11_") for t in tags])[keep]
    b33 = np.array([t.startswith("bt33_") for t in tags])[keep]
    if b11.sum() > 20 and b33.sum() > 20:
        w11 = _ridge_fit(Xb[b11], yb[b11], np.zeros(N_FEATURES), strength=1.0)
        print("bt11->bt33 transfer R^2: %.3f "
              "(train R^2 %.3f)" % (r2(yb[b33], Xb[b33] @ w11),
                                    r2(yb[b11], Xb[b11] @ w11)))
    else:
        print("cross-game check skipped (too few toy samples)")

    save_prior(w, trained_on="toys(bt11,bt33) seeds 0-2 + dev-real failures "
                            "seed 0, n=%d" % len(Xb))
    print("prior saved -> agents/b/success_prior.json")


if __name__ == "__main__":
    main()
