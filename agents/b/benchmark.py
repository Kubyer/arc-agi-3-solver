"""Benchmark Builder B agent (object-centric world model + curiosity).

Runs run_episode on each game and prints per-game: levels completed,
actions taken, per-level RHAE-style scores vs metadata baselines, and the
level-index-weighted game score (same formula as arc_agi/scorecard.py).

Usage: .venv/bin/python agents/b/benchmark.py [game_id ...]
       (default: all 25 real games + bt11/bt33)
       GOALS_OFF=1 disables the goal-hypothesis module (pre-change baseline).
Results are also written to agents/b/RESULTS.md.
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from b.agent import run_episode  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REAL_DIR = os.path.join(CODE, "environment_files")
TOY_DIR = os.path.join(CODE, "ARC-AGI", "test_environment_files")

ALL_REAL = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
            "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
            "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
            "wa30"]
TOYS = ["bt11", "bt33"]


def find_metadata(game_id, env_dir):
    matches = []
    for meta in glob.glob(os.path.join(env_dir, "**", "metadata.json"),
                          recursive=True):
        with open(meta) as f:
            md = json.load(f)
        gid = md.get("game_id", "")
        if gid.split("-")[0] == game_id:
            matches.append((os.path.getmtime(meta), md))
    if not matches:
        raise FileNotFoundError("no metadata for %s" % game_id)
    # client.make() with no version picks the most recently downloaded
    # version; approximate with newest metadata mtime (matters for r11l,
    # which has two local versions).
    matches.sort(key=lambda t: t[0], reverse=True)
    return matches[0][1]


def level_rhae(baseline, actions):
    if actions <= 0:
        return 0.0
    return min(((baseline / actions) ** 2) * 100.0, 115.0)


def weighted_score(level_scores, win_levels):
    """level_scores: {idx: score}. Same formula as scorecard.to_score():
    weights are 1-based level indices (level 0 has weight 1)."""
    total = wsum = maxw = 0.0
    for i in range(win_levels):
        w = i + 1
        s = level_scores.get(i, 0.0)
        total += s * w
        wsum += w
        if s > 0:
            maxw += w
    if wsum == 0:
        return 0.0
    return min(total / wsum, maxw / wsum * 100.0)


def run_game(client, game_id, env_dir, max_actions, max_wall_s, seed):
    md = find_metadata(game_id, env_dir)
    baselines = md.get("baseline_actions", [])
    env = client.make(game_id=game_id, seed=seed)
    if env is None:
        return {"game_id": game_id, "error": "make() returned None"}
    t0 = time.time()
    obs0 = env.reset()
    true_win = getattr(obs0, "win_levels", 0) or 0
    res = run_episode(env, max_actions=max_actions, max_wall_s=max_wall_s,
                      seed=seed, baselines=baselines, verbose=False,
                      enable_goals=os.environ.get("GOALS_OFF") != "1")
    res["game_id"] = game_id
    res["title"] = md.get("title", game_id)
    res["tags"] = md.get("tags", [])
    res["baselines"] = baselines
    # authoritative level count comes from the observation, not the env
    res["win_levels"] = true_win or len(baselines)
    res["elapsed"] = time.time() - t0
    return res


def report(res):
    gid = res["game_id"]
    if "error" in res:
        return "%-6s ERROR: %s" % (gid, res["error"]), None
    n = res["levels_completed"]
    baselines = res["baselines"]
    # win_levels: infer from baselines length if unknown
    win_levels = res.get("win_levels") or len(baselines)
    lv_scores = {}
    lv_detail = []
    for i in range(n):
        b = baselines[i] if i < len(baselines) else (baselines[-1] if baselines else 60)
        a = res["level_actions"].get(i, 0)
        s = level_rhae(b, a)
        lv_scores[i] = s
        lv_detail.append("L%d:%d/%d=%.1f" % (i, a, b, s))
    w = weighted_score(lv_scores, win_levels)
    goal = res.get("goal") or {}
    gstat = ""
    if goal.get("tests"):
        gstat = " gt=%d gc=%d" % (goal["tests"], goal["confirmed"])
    line = ("%-6s %-4s lv=%d/%d act=%d go=%d ret=%d arch=%d wall=%ds "
            "W-RHAE=%.1f%s | %s" % (
                gid, ",".join(res["tags"]) or "-",
                n, win_levels, res["actions"], res["game_overs"],
                res["returns"], res["archive_size"], res["wall_s"], w,
                gstat, " ".join(lv_detail)))
    return line, {"game_id": gid, "levels": n, "win_levels": win_levels,
                  "score": w, "actions": res["actions"],
                  "detail": lv_detail, "stop": res["stop"]}


def main():
    only = sys.argv[1:]
    games = []
    for gid in (only or TOYS + ALL_REAL):
        env_dir = TOY_DIR if gid in TOYS else REAL_DIR
        games.append((gid, env_dir))
    clients = {}
    summary = []
    lines = []
    for gid, env_dir in games:
        if env_dir not in clients:
            clients[env_dir] = Arcade(operation_mode=OperationMode.OFFLINE,
                                      environments_dir=env_dir)
        print("=== %s ===" % gid, flush=True)
        res = run_game(clients[env_dir], gid, env_dir,
                       max_actions=4000, max_wall_s=180, seed=0)
        line, summ = report(res)
        print(line, flush=True)
        lines.append(line)
        if summ:
            summary.append(summ)
    total_score = sum(s["score"] for s in summary) / max(1, len(summary))
    total_lv = sum(s["levels"] for s in summary)
    print("\nMEAN W-RHAE over %d games: %.1f | total levels: %d"
          % (len(summary), total_score, total_lv))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "RESULTS.md"), "w") as f:
        f.write("# Builder B benchmark results\n\n")
        f.write("Agent: object-centric world model + curiosity exploration "
                "(Go-Explore). Offline, CPU-only, numpy/scipy. "
                "Seed 0, max 4000 counted actions / 180s wall per game.\n\n")
        f.write("Per-level RHAE = min(((baseline/actions)^2)*100, 115); "
                "W-RHAE = level-index-weighted mean, 1-based weights (official formula).\n\n")
        f.write("```\n" + "\n".join(lines) + "\n```\n\n")
        f.write("MEAN W-RHAE: %.1f | total levels completed: %d\n" %
                (total_score, total_lv))


if __name__ == "__main__":
    main()
