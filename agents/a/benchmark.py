"""Benchmark the MDL agent on ARC-AGI-3 games.

Usage:  .venv/bin/python benchmark.py [game_id ...] [--max-actions N] [--time-budget S]
Prints per-game: levels completed, actions, per-level RHAE vs baselines.
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_agi import Arcade, OperationMode
from mdl_agent import MDLAgent
from world_model import rhae

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ENV_DIRS = [os.path.join(CODE, "environment_files"),
            os.path.join(CODE, "ARC-AGI", "test_environment_files")]


def _find_meta(game_short):
    """Return (env_dir, metadata_dict) for a game id or id-prefix."""
    for env_dir in ENV_DIRS:
        metas = glob.glob(os.path.join(env_dir, "**", "metadata.json"),
                          recursive=True)
        for m in metas:
            try:
                with open(m) as f:
                    md = json.load(f)
            except Exception:
                continue
            gid = md.get("game_id", "")
            if gid == game_short or gid.startswith(game_short + "-"):
                return env_dir, md
    return None, None


def baselines_for(game_short):
    env_dir, md = _find_meta(game_short)
    if md and "baseline_actions" in md:
        return md["baseline_actions"]
    return None


def env_dir_for(game_short):
    env_dir, _ = _find_meta(game_short)
    return env_dir or ENV_DIRS[0]


def rhae_level_scores(level_actions, baselines, levels_completed):
    """Per-level RHAE: min((baseline/actions)^2*100, 115) for completed levels."""
    scores = []
    for i, b in enumerate(baselines):
        if i < levels_completed:
            scores.append(rhae(level_actions.get(i, 1), b))
        else:
            scores.append(0.0)
    return scores


def env_score(level_scores):
    """Level-index-weighted average, capped like the official scorecard."""
    total = sum(s * i for i, s in enumerate(level_scores))
    weights = sum(range(len(level_scores)))
    if weights == 0:
        return 0.0
    score = total / weights
    max_w = sum(i for i, s in enumerate(level_scores) if s > 0)
    return min(score, max_w / weights * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=None)
    ap.add_argument("--max-actions", type=int, default=350)
    ap.add_argument("--time-budget", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    games = args.games or sorted(
        d for d in os.listdir(ENV_DIRS[0])
        if os.path.isdir(os.path.join(ENV_DIRS[0], d)))

    rows = []
    clients = {}
    for gid in games:
        baselines = baselines_for(gid)
        if baselines is None:
            print(f"{gid}: no metadata, skipped")
            continue
        env_dir = env_dir_for(gid)
        if env_dir not in clients:
            clients[env_dir] = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=env_dir)
        client = clients[env_dir]
        env = client.make(game_id=gid, seed=args.seed)
        if env is None:
            print(f"{gid}: make() failed, skipped")
            continue
        agent = MDLAgent(lam=args.lam, seed=args.seed,
                         max_actions=args.max_actions,
                         time_budget=args.time_budget,
                         verbose=args.verbose)
        t0 = time.time()
        stats = agent.run_episode(env)
        dt = time.time() - t0
        lv = rhae_level_scores(stats["level_actions"], baselines,
                               stats["levels_completed"])
        es = env_score(lv)
        rows.append((gid, stats, lv, es, dt))
        lv_str = ",".join(f"{s:.0f}" for s in lv[:stats["levels_completed"]])
        print(f"{gid:6s} levels {stats['levels_completed']:2d}/{stats['win_levels']:2d} "
              f"actions {stats['actions']:4d} levelRHAE [{lv_str}] "
              f"envScore {es:6.2f} wins {stats['wins']} go {stats['gameovers']} "
              f"keys {stats['keys_learned']} refits {stats['refits']} "
              f"{dt:5.1f}s state {stats['final_state']}", flush=True)

    print("\n=== SUMMARY ===")
    tot = sum(r[3] for r in rows)
    print(f"games={len(rows)} mean_env_score={tot / max(1, len(rows)):.2f} "
          f"total_levels={sum(r[1]['levels_completed'] for r in rows)}")
    for gid, stats, lv, es, dt in rows:
        print(f"  {gid}: score={es:.2f} levels={stats['levels_completed']}/"
              f"{stats['win_levels']} actions={stats['actions']}")


if __name__ == "__main__":
    main()
