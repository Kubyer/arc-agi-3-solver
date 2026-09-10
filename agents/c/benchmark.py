"""Benchmark the causal world-model agent on ARC-AGI-3 games.

Runs run_episode() per game, then computes official-style RHAE scores from
the raw scorecard:
    level_score[i] = min(((baseline[i]/actions[i])^2)*100, 115)
    env_score = min( sum((i+1)*score[i])/sum(i+1),
                     100 * sum(i+1 for completed)/sum(i+1) )
exactly replicating EnvironmentScoreCalculator.

Usage:
    cd code && .venv/bin/python agents/c/benchmark.py [--games lf52,g50t] [--seed 0]
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcengine import GameState  # noqa: F401
from arc_agi import Arcade, OperationMode

from agent import run_episode

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ENV_DIRS = [
    os.path.join(CODE_DIR, "environment_files"),            # 25 real games
    os.path.join(CODE_DIR, "ARC-AGI", "test_environment_files"),  # bt11/bt33
]


def load_metadata():
    meta = {}
    for d in ENV_DIRS:
        pats = [os.path.join(d, "*", "metadata.json"),
                os.path.join(d, "*", "*", "metadata.json")]
        for pat in pats:
            for f in glob.glob(pat):
                m = json.load(open(f))
                gid = m.get("game_id", "")
                short = gid.split("-")[0]
                # prefer deeper (versioned) entries on collision
                if short not in meta or f.count(os.sep) > meta[short][1]:
                    meta[short] = (m, f.count(os.sep))
    return {k: v[0] for k, v in meta.items()}


def raw_card_for(client, game_id, guid):
    mgr = client.scorecard_manager
    for sc in mgr.scorecards.values():
        try:
            idx = sc.cards[game_id].index_of_guid(guid)
        except (KeyError, ValueError):
            continue
        return sc.cards[game_id], idx
    # fallback: any card whose key starts with game_id
    for sc in mgr.scorecards.values():
        for key, card in sc.cards.items():
            if key.startswith(game_id + "-") or key == game_id:
                try:
                    return card, card.index_of_guid(guid)
                except ValueError:
                    pass
    return None, None


def compute_scores(card, idx, baselines):
    """Replicate EnvironmentScoreCalculator from raw card data."""
    abl = card.actions_by_level[idx]  # [(levels_completed, cum_actions)]
    level_scores, level_actions = [], []
    prev = 0
    for lvl, cum in abl:
        a = cum - prev
        prev = cum
        b = baselines[lvl - 1] if 0 < lvl <= len(baselines) else None
        s = min(((b / a) ** 2) * 100.0, 115.0) if (a > 0 and b) else 0.0
        level_scores.append(round(s, 1))
        level_actions.append(a)
    total = wsum = maxw = 0.0
    for i in range(len(baselines)):
        w = i + 1
        s = level_scores[i] if i < len(level_scores) else 0.0
        total += s * w
        wsum += w
        if s > 0:
            maxw += w
    env = min(total / wsum, 100.0 * maxw / wsum) if wsum else 0.0
    return level_scores, level_actions, round(env, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=None,
                    help="comma-separated short game ids (default: all 25+2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = load_metadata()
    games = sorted(meta.keys())
    if args.games:
        want = [g.strip() for g in args.games.split(",")]
        games = [g for g in games if g in want]
    # put toy games last
    games = [g for g in games if g not in ("bt11", "bt33")] + \
            [g for g in games if g in ("bt11", "bt33")]

    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(CODE_DIR, "environment_files"))
    toy_client = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=os.path.join(CODE_DIR, "ARC-AGI",
                                     "test_environment_files"))
    results = []
    t0 = time.time()
    for gid in games:
        m = meta[gid]
        baselines = m.get("baseline_actions", [])
        c = toy_client if gid in ("bt11", "bt33") else client
        env = c.make(game_id=gid, seed=args.seed)
        if env is None:
            print(f"{gid}: make() FAILED", flush=True)
            continue
        stats = run_episode(env, gid, baselines, seed=args.seed,
                            max_actions=args.max_actions)
        guid = None
        card = idx = None
        # recover guid from the env wrapper
        try:
            guid = env._guid
            card, idx = raw_card_for(c, m.get("game_id", gid), guid)
        except Exception as e:
            print(f"{gid}: scorecard lookup failed: {e}")
        if card is not None:
            ls, la, env_score = compute_scores(card, idx, baselines)
        else:
            ls, la, env_score = [], [], 0.0
        nlevels = len(baselines)
        res = {
            "game": gid,
            "title": m.get("title"),
            "tags": m.get("tags"),
            "levels_completed": stats.get("levels_completed", 0),
            "n_levels": nlevels,
            "actions": stats.get("actions", 0),
            "resets": stats.get("resets", 0),
            "level_scores": ls,
            "level_actions": la,
            "baselines": baselines,
            "env_score": env_score,
            "seconds": stats.get("seconds"),
            "active_model": stats.get("active_model"),
            "n_samples": stats.get("n_samples"),
        }
        results.append(res)
        print(f"{gid:5s} | lv {res['levels_completed']:2d}/{nlevels:2d} | "
              f"act {res['actions']:5d} | rst {res['resets']:3d} | "
              f"score {env_score:6.1f} | lv_scores {ls} | "
              f"{res['seconds']}s model={res['active_model']}", flush=True)

    dt = time.time() - t0
    mean_score = (sum(r["env_score"] for r in results) / len(results)
                  if results else 0.0)
    tot_lv = sum(r["levels_completed"] for r in results)
    print(f"\n{len(results)} games, {tot_lv} levels completed, "
          f"mean env score {mean_score:.1f}, {dt:.0f}s total")

    out = args.out or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "RESULTS.json")
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
