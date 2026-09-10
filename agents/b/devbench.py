"""Dev-subset benchmark for agent B planner iteration (workstream 1).

Usage: .venv/bin/python agents/b/devbench.py [seed...]
Runs ar25 g50t lp85 lf52 vc33 at each seed, prints compact lines:
  gid seed lv=X/Y lvmax=M act=N wall=S go=G ret=R gt=T gc=C gp=P fga=F ev=A,B stop=...

lvmax = max levels_completed ever observed (diagnostic: the final count can
return to 0 after later resets); ev = action indices of completion events;
fga = first action index where a goal key won (None when GOALS_OFF).

Compact per-game results go to agents/b/hidden_files/devbench_<tag>.log when
DEV_TAG is set (e.g. DEV_TAG=base_off).
"""
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
    games = sys.argv[1:] if len(sys.argv) > 1 and not sys.argv[1].isdigit() \
        else DEV
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0]
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=REAL_DIR)
    lines = []
    for gid in games:
        for seed in seeds:
            t0 = time.time()
            res = run_game(client, gid, REAL_DIR, max_actions=4000,
                           max_wall_s=180, seed=seed)
            goal = res.get("goal") or {}
            evts = res.get("level_events") or []
            ev_str = ",".join(str(e["actions"]) for e in evts) or "-"
            line = ("%-6s seed=%d lv=%d/%d lvmax=%d act=%d wall=%ds go=%d "
                    "ret=%d gt=%d gc=%d gp=%d fga=%s ev=%s stop=%s" % (
                        gid, seed, res["levels_completed"],
                        res.get("win_levels", 0), res.get("levels_ever", 0),
                        res["actions"],
                        int(res["wall_s"]), res["game_overs"],
                        res["returns"], goal.get("tests", 0),
                        goal.get("confirmed", 0), goal.get("pursuits", 0),
                        str(res.get("first_goal_action")),
                        ev_str, res["stop"]))
            print(line, flush=True)
            lines.append(line)
    tag = os.environ.get("DEV_TAG")
    if tag:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "hidden_files"), exist_ok=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "hidden_files",
                                 "devbench_%s.log" % tag), "w") as f:
            f.write("goals=%s %s\n" %
                    ("off" if os.environ.get("GOALS_OFF") == "1" else "on",
                     time.strftime("%Y-%m-%d %H:%M UTC")))
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
