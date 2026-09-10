"""Prediction-error evaluation for the MDL graft (workstream 2).

Usage: .venv/bin/python agents/b/mdl_eval.py [seed...]
Runs the dev subset (ar25 g50t lp85 lf52 vc33) with MDL_EVAL=1 and prints,
per game/seed, the grid-space Hamming errors (identity / MDL program /
rendered B-stats) and signature-prediction accuracies, plus the learned
per-key rules. Behavior-neutral: the eval hook uses pure functions only.

Logs go to agents/b/hidden_files/mdl_eval_<tag>.log when DEV_TAG is set.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from b.benchmark import run_game  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REAL_DIR = os.path.join(CODE, "environment_files")
DEV = ["ar25", "g50t", "lp85", "lf52", "vc33"]


def main():
    os.environ["MDL_EVAL"] = "1"
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0]
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=REAL_DIR)
    lines = []
    for gid in DEV:
        for seed in seeds:
            res = run_game(client, gid, REAL_DIR, max_actions=4000,
                           max_wall_s=180, seed=seed)
            ev = res.get("mdl_eval") or {}
            mdl = res.get("mdl") or {}
            line = ("%-6s seed=%d n=%d id_err=%.1f mdl_err=%.1f(n=%d%s) "
                    "stats_err=%.1f sigacc=id:%.2f/mdl:%.2f/stats:%.2f "
                    "csigacc=id:%.2f/mdl:%.2f "
                    "rules=%d fals=%d lvmax=%d wall=%ds" % (
                        gid, seed, ev.get("n", 0),
                        ev.get("identity_mean", -1),
                        ev.get("mdl_mean", -1), ev.get("mdl_n", 0),
                        ("/mat:%.1f(n=%d)" % (ev.get("mdl_mature_mean", -1),
                                              ev.get("mdl_mature_n", 0))
                         if ev.get("mdl_mature_n") else ""),
                        ev.get("stats_mean", -1),
                        ev.get("sig_acc_identity", -1),
                        ev.get("sig_acc_mdl", -1),
                        ev.get("sig_acc_stats", -1),
                        ev.get("sig_cacc_identity", -1),
                        ev.get("sig_cacc_mdl", -1),
                        len(mdl.get("rules", {})),
                        mdl.get("n_falsified", 0),
                        res.get("levels_ever", 0),
                        int(res["wall_s"])))
            print(line, flush=True)
            lines.append(line)
            for k, r in sorted(mdl.get("rules", {}).items()):
                rl = "    %-4s %s" % (k, r)
                print(rl, flush=True)
                lines.append(rl)
    tag = os.environ.get("DEV_TAG")
    if tag:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "hidden_files"), exist_ok=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "hidden_files",
                                 "mdl_eval_%s.log" % tag), "w") as f:
            f.write("goals=%s %s\n" %
                    ("off" if os.environ.get("GOALS_OFF") == "1" else "on",
                     time.strftime("%Y-%m-%d %H:%M UTC")))
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
