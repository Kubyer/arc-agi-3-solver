"""Paired goals-on / GOALS_OFF lf52 probes for max-level attribution.

Usage: .venv/bin/python agents/b/probe_lf52.py [seed...]
Runs lf52 with goals on and with GOALS_OFF=1 (same 4000-action / 180s
budgets, same seeds) and prints, per run:
  - levels_ever (max obs.levels_completed ever seen), final levels
  - action index + goal snapshot of every level-completion event
  - first action index where a goal key won (goals-on only)

This answers: does the goal planner cause lf52 L0 completions, or does
baseline curiosity also complete it before a later reset wipes the
final count?
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


def run_one(client, seed, goals_off):
    if goals_off:
        os.environ["GOALS_OFF"] = "1"
    else:
        os.environ.pop("GOALS_OFF", None)
    t0 = time.time()
    res = run_game(client, "lf52", REAL_DIR, max_actions=4000,
                   max_wall_s=180, seed=seed)
    evts = res.get("level_events") or []
    print("lf52 seed=%d goals=%s final_lv=%d lvmax=%d act=%d wall=%ds "
          "go=%d fga=%s stop=%s" % (
              seed, "off" if goals_off else "on",
              res["levels_completed"], res.get("levels_ever", 0),
              res["actions"], int(time.time() - t0), res["game_overs"],
              res.get("first_goal_action"), res["stop"]), flush=True)
    for e in evts:
        print("    event: actions=%d %d->%d goal=%s" % (
            e["actions"], e["from"], e["to"],
            json.dumps(e["goal"], sort_keys=True)), flush=True)


def main():
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0, 1, 2]
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=REAL_DIR)
    for seed in seeds:
        for goals_off in (False, True):
            run_one(client, seed, goals_off)


if __name__ == "__main__":
    main()
