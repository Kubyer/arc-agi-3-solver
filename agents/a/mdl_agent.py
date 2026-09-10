"""MDL agent: self-falsifying program world-models + MPC planning.

Entry point:  run_episode(env, ...)  where env = client.make(game_id, seed=..).

Loop:
  1. observe frame -> coarsen to 32x32
  2. choose action: epsilon-greedy over {planner beam search, novelty-greedy,
     uniform random}
  3. step env; feed every returned frame as a (s, s') transition into the
     action key's KeyModel (falsification + evolution happen inside)
  4. detect level completion via levels_completed increments -> win memory;
     detect GAME_OVER -> danger statistics -> free RESET
"""

import time
import numpy as np
from arcengine import GameAction, GameState

from world_model import (KeyModel, coarsen, frame_hash16, mode_color,
                         rhae, LAM_DEFAULT)
from planner import Planner


def click_region(qx, qy):
    return (6, qx // 4, qy // 4)  # 8x8 coarse regions of click space


def key_of(aid, data):
    if aid == 6 and data:
        return click_region(int(data["x"]) // 2, int(data["y"]) // 2)
    return aid


class MDLAgent:
    def __init__(self, lam=LAM_DEFAULT, seed=0, max_actions=350,
                 time_budget=240.0, verbose=False):
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.max_actions = max_actions
        self.time_budget = time_budget
        self.verbose = verbose
        self.keymodels = {}
        # global click model: fallback for click regions with little data
        self.click_any = KeyModel(lam=lam, seed=seed + 999)
        self.planner = Planner(seed=seed + 1)
        self.eps = 0.35
        self.burst = None  # (remaining, aid, data): exploration burst state
        self.replay = []   # winning action suffix from a past level, to retry

    def km(self, key):
        m = self.keymodels.get(key)
        if m is None:
            fb = self.click_any if isinstance(key, tuple) else None
            m = KeyModel(lam=self.lam, seed=int(self.rng.integers(1 << 30)),
                         fallback=fb)
            self.keymodels[key] = m
        return m

    def choose(self, g32, legal_ids):
        # ongoing exploration burst: repeat the same action
        if self.burst is not None:
            n, aid, data = self.burst
            if aid in legal_ids:
                self.burst = (n - 1, aid, data) if n > 1 else None
                return aid, data
            self.burst = None
        # replay a past winning trajectory: levels often share mechanics
        if self.replay and self.rng.random() < 0.30:
            aid, data = self.replay.pop(0)
            if aid in legal_ids:
                return aid, data
        r = self.rng.random()
        if r < 0.10:
            # start a burst: repeat one action 4-10 times. This is how the
            # agent discovers sustained effects (corridors, repeated clicks)
            # that 1-step novelty cannot see.
            aid = int(self.rng.choice(legal_ids))
            data = None
            if aid == 6:
                # burst at a promising candidate, not a uniform-random pixel
                cands = self.planner.click_candidates(
                    g32, mode_color(g32))
                qx, qy = cands[int(self.rng.integers(len(cands)))] \
                    if cands else (int(self.rng.integers(32)),
                                   int(self.rng.integers(32)))
                data = {"x": qx * 2 + 1, "y": qy * 2 + 1}
            self.burst = (int(self.rng.integers(4, 11)), aid, data)
            return aid, data
        if r < 0.03 + 0.10:  # uniform random discovery
            aid = int(self.rng.choice(legal_ids))
            data = None
            if aid == 6:
                qx, qy = int(self.rng.integers(32)), int(self.rng.integers(32))
                data = {"x": qx * 2 + 1, "y": qy * 2 + 1}
            return aid, data
        if r < self.eps + 0.13:  # novelty-greedy exploration
            act = self.planner.greedy_novelty_action(
                g32, legal_ids, self.keymodels, key_of)
            if act is not None:
                return act
        act = self.planner.plan(g32, legal_ids, self.keymodels, key_of)
        if act is not None:
            return act
        # last resort: uniform random legal action (never return None)
        aid = int(self.rng.choice(legal_ids))
        data = None
        if aid == 6:
            qx, qy = int(self.rng.integers(32)), int(self.rng.integers(32))
            data = {"x": qx * 2 + 1, "y": qy * 2 + 1}
        return aid, data

    def run_episode(self, env):
        t0 = time.time()
        obs = env.reset()
        prev_g32 = None
        fresh = True  # no transition to record from this frame yet
        levels_prev = obs.levels_completed
        win_levels = obs.win_levels
        actions = 0
        level_actions = {}
        cur_level = 0
        wins = 0
        gameovers = 0
        last_key = None
        last_click = None  # (qx, qy) on 32-grid for ACTION6, else None
        pre_g32 = None
        acts_since_win = 0
        traj = []  # (aid, data) of the current level, for win replay

        while True:
            if obs.state is GameState.WIN:
                break
            if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                if obs.state is GameState.GAME_OVER and last_key is not None:
                    self.km(last_key).note_gameover()
                    gameovers += 1
                    self.eps = min(0.6, self.eps + 0.08)
                obs = env.step(GameAction.RESET)
                fresh = True
                last_key = None
                self.burst = None  # don't continue a burst across a reset
                traj = []  # the failed attempt is not worth replaying
                continue

            if actions >= self.max_actions:
                break
            if time.time() - t0 > self.time_budget:
                break

            legal = [int(i) for i in obs.available_actions if int(i) != 0]
            if not legal:
                # no legal actions: reset rather than crash
                env.step(GameAction.RESET)
                obs = env.reset()
                fresh = True
                continue
            frames = obs.frame
            g32 = coarsen(frames[-1]) if frames else None
            if g32 is None:
                obs = env.step(GameAction.RESET)
                fresh = True
                continue

            # --- learn from the transition that led here -------------------
            if not fresh and prev_g32 is not None and last_key is not None:
                seq = [prev_g32] + [coarsen(f) for f in frames]
                km = self.km(last_key)
                for a, b in zip(seq[:-1], seq[1:]):
                    if np.array_equal(a, b):
                        # no visible change: still informs prediction error
                        km.note_prediction(a, b, click=last_click)
                        continue
                    km.observe(a, b, click=last_click)
                    if isinstance(last_key, tuple):
                        # share click data with the global click model
                        self.click_any.observe(a, b, click=last_click)
            # --- level completion signal ----------------------------------
            if obs.levels_completed > levels_prev:
                wins += 1
                if pre_g32 is not None and last_key is not None:
                    self.planner.note_win(last_key, frame_hash16(pre_g32))
                # remember how we won, to replay on later (similar) levels
                if traj:
                    self.replay = [t for t in traj[-80:]]
                traj = []
                self.planner.on_level_start()
                self.eps = max(0.08, self.eps * 0.5)
                levels_prev = obs.levels_completed
                acts_since_win = 0
                cur_level = obs.levels_completed
            self.planner.note_frame(g32)

            # --- act -------------------------------------------------------
            aid, data = self.choose(g32, legal)
            if aid is None:
                aid, data = legal[0], None
            key = key_of(aid, data)
            self.km(key).tries += 1
            pre_g32 = g32
            if aid == 6 and data is not None:
                qx, qy = int(data["x"]) // 2, int(data["y"]) // 2
            else:
                qx = qy = None

            obs = env.step(GameAction.from_id(aid), data=data,
                           reasoning=None)
            actions += 1
            acts_since_win += 1
            level_actions[cur_level] = level_actions.get(cur_level, 0) + 1
            traj.append((aid, data))

            # click-effect bookkeeping needs the post frame
            if qx is not None and obs.frame:
                post = coarsen(obs.frame[-1])
                changed = not np.array_equal(pre_g32, post)
                self.planner.note_click_effect(qx, qy, changed)
                if not changed:
                    self.planner.note_click_dead(key)

            if acts_since_win > 150:
                self.eps = max(self.eps, 0.5)

            prev_g32 = g32
            last_key = key
            last_click = (qx, qy) if qx is not None else None
            fresh = False

            if self.verbose and actions % 50 == 0:
                print(f"  ... {actions} actions, levels={obs.levels_completed}, "
                      f"eps={self.eps:.2f}, keys={len(self.keymodels)}",
                      flush=True)

        total_refits = sum(m.refits for m in self.keymodels.values())
        try:
            md = env.metadata() if hasattr(env, "metadata") else {}
            baselines = (md.get("action_baselines")
                         or md.get("baselines") or [])
        except Exception:
            baselines = []
        return {
            "levels_completed": obs.levels_completed,
            "win_levels": win_levels,
            "won": obs.state is GameState.WIN,
            "actions": actions,
            "level_actions": level_actions,
            "level_rhae": [rhae(level_actions.get(i, 0), baselines[i])
                           if i < len(baselines) else None
                           for i in range(obs.levels_completed)],
            "wins": wins,
            "gameovers": gameovers,
            "keys_learned": len(self.keymodels),
            "refits": total_refits,
            "time_s": time.time() - t0,
            "final_state": str(obs.state),
        }


def run_episode(env, **kwargs):
    """Clean entry point: run one episode, return stats dict."""
    return MDLAgent(**kwargs).run_episode(env)
