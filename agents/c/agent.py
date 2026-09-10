"""Causal world-model agent: actions are interventions in a learned SCM.

Episode loop
------------
* Parse each frame into binned causal variables (variables.py).
* Enumerate interventions: each available simple action id, plus ACTION6
  parameterized by clicked-object identity  ('C', slot) = click the topmost
  pixel of that color slot, ('CB',) = click background (null control).
* Every step is recorded as interventional data
  (pre-vars, do(I), post-vars, context, outcome) in the CausalModel.
* Control = greedy 1-step simulation on the SCM toward the learned
  win-direction (planner.py); curiosity-driven exploration before the first
  win; beam search over the SCM when stalled.

RESET safety (verified against arcengine internals):
  game._action_count == 0  =>  RESET triggers a FULL reset (score wiped).
  This happens (a) right after env.reset(), (b) after a level win the *next*
  action loads the new level and leaves _action_count == 0 again. We track
  this exactly and only RESET when mandatory (GAME_OVER) -- RESETs also
  count toward the scorecard action total, so voluntary resets are avoided.

Entry point: run_episode(env, game_id, baselines, ...) -> stats dict.
"""

import random
import time
from collections import deque

from arcengine import GameAction, GameState

from variables import FrameParser, VAR_NAMES
from causal import CausalModel, _signed_delta
from planner import Planner

RESET = GameAction.RESET


def _label_action(label):
    kind = label[0]
    if kind == "A":
        return GameAction.from_id(label[1]), None
    if kind == "C":
        _, slot, x, y = label
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}
    if kind == "CB":
        _, x, y = label
        return GameAction.ACTION6, {"x": int(x), "y": int(y)}
    raise ValueError(label)


class CausalAgent:
    def __init__(self, seed=0, max_actions=4000, verbose=False):
        self.parser = FrameParser()
        self.model = CausalModel(seed=seed)
        self.planner = Planner(self.model, seed=seed)
        self.rng = random.Random(seed)
        self.max_actions = max_actions
        self.verbose = verbose
        self._stall_hist = deque(maxlen=14)
        self._last_labels = deque(maxlen=10)
        self._recent = deque(maxlen=25)  # recent pre-states (doom marking)

    # ------------------------------------------------------- interventions
    def enumerate_interventions(self, obs, parsed):
        labels = []
        avail = obs.available_actions or []
        for aid in avail:
            if aid == 0:
                continue
            if aid == 6:
                continue
            labels.append(("A", aid))
        if 6 in avail:
            meta = parsed["meta"]
            for slot, color in sorted(meta["slot_color"].items()):
                top = meta["top"].get(slot)
                if top is None:
                    continue
                y, x = top
                labels.append(("C", slot, int(x), int(y)))
            bgp = meta["bg_pixel"]
            if bgp is not None:
                y, x = bgp
                labels.append(("CB", int(x), int(y)))
        return labels

    # ------------------------------------------------------------------ run
    def run_episode(self, env, game_id, baselines, seed=0):
        t0 = time.time()
        obs = env.reset()
        if obs is None:
            return {"error": "make/reset failed"}

        self.parser.reset_slots()
        # RESET-safety bookkeeping (mirrors game._action_count)
        zero = True            # True => RESET would FULL-reset (score wipe)
        pending_load = False   # True => next real action loads the new level

        prev = None            # parsed previous observation
        prev_levels = 0
        total = 0              # non-reset steps total
        resets = 0
        wins = 0
        deaths = 0
        forced_explore = 0

        def parse_obs(o):
            if o is None or len(o.frame) == 0:
                return None
            return self.parser.parse(o.frame[-1])

        prev = parse_obs(obs)
        if prev is None:
            return {"error": "empty initial frame"}
        level_start_bins = dict(prev["bins"])  # vars at level-load moment
        # per-LEVEL action budget (across attempts: deaths don't reset it,
        # because failed attempts' actions count toward the level's score)
        lvl_spent = 0

        level_cap = self._level_cap(baselines, 0)
        last_win_sig = None

        while total < self.max_actions:
            st = obs.state
            if st == GameState.WIN:
                wins = obs.levels_completed
                break
            if st == GameState.NOT_PLAYED:
                obs = env.step(RESET)
                resets += 1
                zero = True
                pending_load = False
                prev = parse_obs(obs)
                prev_levels = obs.levels_completed if obs else 0
                continue
            if st == GameState.GAME_OVER:
                # mandatory reset (level_reset: _action_count -> 0)
                obs = env.step(RESET)
                resets += 1
                deaths += 1
                zero = True
                pending_load = False
                prev = parse_obs(obs)
                prev_levels = obs.levels_completed if obs else 0
                if prev is not None:
                    level_start_bins = dict(prev["bins"])
                self._recent.clear()
                self._stall_hist.clear()
                continue

            # ---- level completed? (win signature -> goal direction)
            lv = obs.levels_completed
            if lv > prev_levels:
                wins = lv
                # NOTE: the winning transition was already recorded as a
                # sample with outcome='win' right after that step (below).
                prev_levels = lv
                lvl_spent = 0
                pending_load = True
                level_cap = self._level_cap(baselines, lv)
                self._stall_hist.clear()
                if self.verbose:
                    print(f"[{game_id}] level {lv} complete "
                          f"(model={self.model.active})")

            if lvl_spent >= level_cap:
                if self.verbose:
                    print(f"[{game_id}] level cap hit ({level_cap})")
                break

            cur = parse_obs(obs)
            if cur is None:  # should not happen in NOT_FINISHED
                obs = env.step(RESET)
                resets += 1
                continue
            ctx = (obs.levels_completed, cur["meta"]["presence"])

            labels = self.enumerate_interventions(obs, cur)
            if not labels:
                break

            # ---- choose intervention
            action_label = self._choose(cur, ctx, labels)
            if action_label is None:
                # stall fallback: least-tried among non-lethal candidates
                safe = self.planner.safe_candidates(
                    cur["bins"], ctx, [self._canon(L) for L in labels])
                pool = [L for L in labels if self._canon(L) in safe] or labels
                action_label = min(pool,
                                   key=lambda L: self.model.n_I.get(
                                       self._canon(L), 0))
                forced_explore += 1

            act, data = _label_action(action_label)
            pre_bins = cur["bins"]
            pre_frame = obs.frame[-1] if len(obs.frame) else None
            self.model.note_seen(pre_bins)
            self._recent.append(dict(pre_bins))
            obs2 = env.step(act, data=data)
            total += 1
            lvl_spent += 1

            # ---- RESET-safety update for the action just taken
            was_load = pending_load
            if pending_load:
                # this was the level-load action: game zeroed _action_count
                pending_load = False
                zero = True
            else:
                zero = False

            # ---- record interventional data
            post = parse_obs(obs2)
            post_frame = obs2.frame[-1] if len(obs2.frame) else None
            if post is not None:
                new_lv = obs2.levels_completed
                if new_lv > lv:
                    outcome = "win"
                elif obs2.state == GameState.GAME_OVER:
                    outcome = "died"
                else:
                    outcome = "none"
                px_changed = int((pre_frame != post_frame).sum()) \
                    if pre_frame is not None and post_frame is not None else 0
                ds = self.model.observe(pre_bins, post["bins"],
                                        self._canon(action_label),
                                        ctx, outcome, px_changed=px_changed)
                if outcome == "died":
                    # credit assignment: the dooming move may be many steps
                    # back; mark the recent trajectory as doomed states.
                    for b in self._recent:
                        self.model.note_doomed(b)
                    self._recent.clear()
                if outcome == "win":
                    # goal direction: level-start config -> pre-win config.
                    # (The post-win frame shows the reloaded level, so the
                    #  raw pre->post delta would point the wrong way.)
                    win_ds = {V: _signed_delta(level_start_bins[V],
                                               pre_bins[V], V)
                              for V in VAR_NAMES}
                    self.planner.on_win(win_ds)
                    last_win_sig = {V: win_ds[V] for V in VAR_NAMES
                                    if win_ds[V] != 0}
                    self._recent.clear()
                if was_load:
                    # the load action just finished: fresh level is visible
                    level_start_bins = dict(post["bins"])
                prev = post
            else:
                # empty frame (e.g. right after GAME_OVER/WIN): keep pre
                prev = cur

            # ---- housekeeping: model selection, stall detection
            if total % 120 == 0 and total > 0:
                self.model.select_model()

            rh = self._raw_hash(cur["raw"])
            self._stall_hist.append(rh)
            self._last_labels.append(self._canon(action_label))
            obs = obs2

        # final model selection + graph export data
        self.model.select_model()
        dt = time.time() - t0
        return {
            "game_id": game_id,
            "levels_completed": obs.levels_completed if obs else 0,
            "win_levels": obs.win_levels if obs else 0,
            "actions": total,
            "resets": resets,
            "deaths": deaths,
            "forced_explore": forced_explore,
            "active_model": self.model.active,
            "n_samples": self.model.n_total,
            "n_wins_learned": self.planner.n_wins,
            "win_signature": last_win_sig,
            "seconds": round(dt, 1),
            "graph": self.model.graph_dict(),
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _canon(label):
        # canonicalize: drop click coordinates (object-anchored identity only)
        if label[0] == "C":
            return ("C", label[1])
        if label[0] == "CB":
            return ("CB",)
        return label

    @staticmethod
    def _level_cap(baselines, level_idx):
        if baselines and level_idx < len(baselines):
            b = baselines[level_idx]
            if level_idx == 0:
                # level 0 buys the causal model + goal direction that
                # transfers to later levels: allow a larger discovery budget
                return min(6 * b + 20, 400)
            return min(4 * b + 10, 600)
        return 400

    @staticmethod
    def _raw_hash(raw):
        return tuple(sorted(raw.items()))

    def _choose(self, cur, ctx, labels):
        canon = [self._canon(L) for L in labels]
        canon_labels = list(dict.fromkeys(canon))  # dedup preserve order
        # map canonical -> full label (first occurrence)
        full_of = {}
        for L, C in zip(labels, canon):
            full_of.setdefault(C, L)

        # stall -> beam search over the SCM
        if (len(self._stall_hist) == self._stall_hist.maxlen
                and len(set(self._stall_hist)) == 1
                and self.planner.has_goal()):
            b = self.planner.beam_search(cur["bins"], ctx, canon_labels)
            if b is not None:
                self._stall_hist.clear()
                return full_of[b]
            # beam found nothing: fall through to forced exploration
            return None

        # NOTE: no "same label repeated" anti-loop here on purpose: repeating
        # one intervention is often the correct policy (e.g. press LEFT 16x).
        # True stalls are detected via unchanged raw state (stall_hist).
        scores = self.planner.greedy_scores(cur["bins"], ctx, canon_labels)
        best = max(canon_labels, key=lambda C: scores[C])
        return full_of[best]


def run_episode(env, game_id, baselines, seed=0, max_actions=4000,
                verbose=False):
    """Clean entry point: run one episode, return stats dict."""
    agent = CausalAgent(seed=seed, max_actions=max_actions, verbose=verbose)
    return agent.run_episode(env, game_id, baselines, seed=seed)
