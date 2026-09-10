"""Object-centric world model + curiosity exploration agent.

Pipeline per step:
  frame -> objects (objects.py) -> signature -> archive (archive.py)
Action selection scores each action-key by:
  * untried-here bonus (explicit curiosity),
  * predicted novelty from the learned transition model (model.py),
  * color valence: colors introduced during completed levels are "good",
    colors seen only in failed levels are "bad" (object-level credit;
    valence is only trusted after the first completed level, so early
    exploration is never biased by a single failure),
  * known-good bonus: the action that solved the previous level,
  * trajectory tabu: action sequences that already led to GAME_OVER are
    softly avoided (their prefixes are penalized, never hard-banned, so
    the agent can retry keys in new contexts and can never deadlock),
  * fatal penalty: keys observed to cause *immediate* GAME_OVER.
Go-Explore return: RESET is score-free, so to revisit a frontier archive
cell we RESET and replay its stored trajectory from the level start.
Planning: bounded BFS over the *learned object model* toward frontier
signatures when the policy is stuck; level completion is detected via
levels_completed increments and the winning trajectory becomes the
level's solution template (known-good action transfer).

ACTION6 clicks are action-keys too: one key per clicked object COLOR
("click the largest object of color c", resolved to its live centroid at
execution time -- color-only so knowledge transfers across levels even when
sprite scales change), plus a few coarse grid clicks for games where empty
space matters.

No per-game hardcoding: action meanings, click targets, good/bad colors
are all discovered online.
"""

import os
import math
import random
import time
from collections import Counter, defaultdict, deque

from arcengine import GameAction, GameState

from .archive import Archive
from .goals import (GOAL_COOLDOWN, GOAL_EVERY, GOAL_K, GOAL_M, MIN_MODEL_KEYS,
                    MIN_MODEL_N, N_HYPS_PER_EVAL, PURSUIT_GAIN_MIN,
                    PURSUIT_PLAN_CAP, GoalModule)
from .mdl_model import MDLWorldModel, coarsen_frame, coarse_sig32
from .model import TransitionModel
from .objects import parse_frame
from .success import SuccessModel, featurize

GRID_CLICKS = [(16, 16), (16, 48), (48, 16), (48, 48)]
MAX_CLICK_CLASSES = 12
STUCK_SCORE = 0.6
MAX_RETURNS = 40
PLAN_DEPTH = 3
MAX_TABU = 300


def run_episode(env, max_actions=4000, max_wall_s=180.0, seed=0,
                baselines=None, verbose=False, enable_goals=True):
    """Entry point. env = object from client.make(...). Returns result dict."""
    agent = ObjCuriosityAgent(env, seed=seed, max_actions=max_actions,
                             max_wall_s=max_wall_s, baselines=baselines,
                             verbose=verbose, enable_goals=enable_goals)
    return agent.run()


class ObjCuriosityAgent:
    def __init__(self, env, seed=0, max_actions=4000, max_wall_s=180.0,
                 baselines=None, verbose=False, enable_goals=True):
        self.env = env
        self.rng = random.Random(seed)
        # Separate RNG stream for goal-hypothesis decisions (planning
        # samples, tie-breaks). Goal code must NEVER consume self.rng:
        # that would perturb the curiosity policy's randomness and change
        # behavior on games where the goal module takes no action.
        self.goal_rng = random.Random("goals-%s" % (seed,))
        self.max_actions = max_actions
        self.max_wall_s = max_wall_s
        self.baselines = list(baselines) if baselines else []
        self.verbose = verbose
        # enable_goals=False reproduces the pre-goal-module behavior exactly
        # (goal-hypothesis mode never triggers, hooks are no-ops).
        self.enable_goals = enable_goals

        self.model = TransitionModel()
        # MDL program world-model graft (workstream 2): per-action-key
        # KeyModels from agents/a/world_model.py (loaded by file path;
        # agents/a untouched). Learning is passive (observe() on every
        # transition); consumption is gated on mature, stable programs
        # (see mdl_model.py) and happens in the model's native coarse /
        # grid space -- never by faking 64px-exact signatures.
        self.mdl = MDLWorldModel(seed=seed)
        if os.environ.get("MDL_EVAL") == "1":
            self.mdl.enable_eval()
        self.archive = Archive()
        self.goals = GoalModule()  # goal-hypothesis mode (see goals.py)
        self.goal_plan = []        # pending planned keys for the active test
        self.goal_plan_is_test = True  # False: sustained pursuit plan (no refute)
        self.goal_plan_is_value = False  # True: template-free V-pursuit
        self.last_goal_test = -10 ** 9
        self.goal_eligible_at = 0
        self._last_interleave = -10 ** 9  # last goal-directed action

        # Learned success signal (round 3, see success.py): V(s) predicting
        # progress toward level completion. Per-game model, cold-started
        # from the offline toy-trained prior; every level completion and
        # game-over refits it (prior-centered L2), so level-0 data informs
        # level-1 targets. SUCCESS_OFF=1 disables it entirely (ablation:
        # pure round-2 behavior). No RNG use anywhere in this machinery --
        # numpy-only, deterministic -- so the toy gate stays bit-identical.
        self.success_off = os.environ.get("SUCCESS_OFF") == "1"
        self.success_dump = os.environ.get("SUCCESS_DUMP")  # dir or None
        self.success = SuccessModel.from_prior()
        self._succ_log = []      # per-attempt [(featvec)] every 4th step
        self._succ_dump_X = []   # flat dump buffers (SUCCESS_DUMP only)
        self._succ_dump_y = []
        # Feature logging runs when the signal is on OR when dumping for
        # offline training (SUCCESS_DUMP with SUCCESS_OFF=1). Online
        # learning/refits only run when the signal is on.

        # object-level credit assignment
        self.color_win = Counter()   # colors introduced during completed levels
        self.color_loss = Counter()  # colors introduced during failed levels
        self.good_colors = set()
        self.bad_colors = set()
        self.have_win = False

        self.solutions = {}          # level_idx -> [action-keys] that solved it
        self.known_good = None       # most common key in last solution
        self.seen_keys = set()

        # per-level episodic state
        self.level_idx = -1
        self.traj = []               # [key] since level start
        self.traj_deltas = []        # [(key, added_colors, removed_colors)]
        self.failed_trajs = set()    # tabu: key-sequences that led to GAME_OVER
        self.consec_fails = 0
        self._start_mode = None      # None | 'levelchange' | 'reset'
        self._start_wait = 0

        self.level_actions = defaultdict(int)  # counted actions per level
        self.level_total = defaultdict(int)    # incl. failed attempts
        self.last_levels = 0
        # diagnostic-only max-level tracking (no behavior change): final
        # last_levels can return to 0 after a later reset/game-over, so
        # keep the maximum ever observed plus the goal state that won it.
        self.ever_levels = 0
        self.level_events = []
        self.first_goal_action = None  # diagnostic: first goal-key action
        self._goal_snapshot = {}
        self.since_progress = 0
        self.stall_limit = 1200

        self.stats = Counter(actions=0, resets=0, game_overs=0,
                             returns=0, return_mismatch=0, plans_used=0,
                             level_cap_resets=0, levels_seen=0,
                             surprises=0)

    # ------------------------------------------------------------------
    # action-keys
    # ------------------------------------------------------------------

    def legal_keys(self, obs, objs):
        keys = []
        for aid in obs.available_actions:
            if aid == 0:
                continue
            if aid == 6:
                # one key per color present (largest object of that color is
                # the click target) -- color-only so the model transfers
                # across levels even when sprite scales change.
                colors = sorted({o["color"] for o in objs},
                                key=lambda c: -max(o["size"] for o in objs
                                                   if o["color"] == c))
                for c in colors[:MAX_CLICK_CLASSES]:
                    keys.append("o%d" % c)
                for i in range(len(GRID_CLICKS)):
                    keys.append("g%d" % i)
            else:
                keys.append("a%d" % aid)
        return keys

    def resolve(self, key, objs):
        """action-key -> (GameAction, data). Clicks resolve to live centroids."""
        if key[0] == "a":
            return GameAction.from_id(int(key[1:])), None
        if key[0] == "o":
            color = int(key[1:])
            cands = [o for o in objs if o["color"] == color]
            if cands:
                o = max(cands, key=lambda o: o["size"])
                x, y = int(round(o["cx"])), int(round(o["cy"]))
            else:
                x = y = 32
            return GameAction.ACTION6, {"x": max(0, min(63, x)),
                                       "y": max(0, min(63, y))}
        if key[0] == "g":
            gx, gy = GRID_CLICKS[int(key[1:])]
            return GameAction.ACTION6, {"x": gx, "y": gy}
        raise ValueError("bad key %r" % key)

    def _click64(self, key, objs):
        """Resolved click coordinates in 64-px for a key, or None for
        keyboard keys. Used to ground MDL click programs (goto mode)."""
        if key[0] not in ("o", "g"):
            return None
        _act, data = self.resolve(key, objs)
        if not data:
            return None
        return (int(data["x"]), int(data["y"]))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self):
        t0 = time.time()
        env = self.env
        obs = env.reset()
        self.stats["resets"] += 1
        self._start_mode = "reset"
        self._start_wait = 0

        while True:
            if self.stats["actions"] >= self.max_actions:
                return self.finish("action_budget", t0)
            if time.time() - t0 >= self.max_wall_s:
                return self.finish("wall_budget", t0)
            if self.since_progress >= self.stall_limit:
                return self.finish("stalled", t0)

            st = obs.state
            if st is GameState.WIN:
                self.last_levels = obs.levels_completed
                return self.finish("win", t0)
            if st is GameState.NOT_PLAYED:
                obs = env.step(GameAction.RESET)
                self.stats["resets"] += 1
                continue
            if st is GameState.GAME_OVER:
                self.on_game_over()
                self.consec_fails += 1
                obs = env.step(GameAction.RESET)
                self.stats["resets"] += 1
                self.traj = []
                self.traj_deltas = []
                self._succ_log = []  # success-signal attempt log
                self.goal_plan = []
                self._start_mode = "reset"
                self._start_wait = 0
                continue

            # ---- NOT_FINISHED ----
            cur = obs.levels_completed
            self.last_levels = cur
            if cur != self.level_idx:
                if self.level_idx >= 0 and cur == self.level_idx + 1:
                    # record the new maximum BEFORE on_level_complete can
                    # mutate goal state; _goal_snapshot was saved at the
                    # decision that issued the winning action.
                    if cur > self.ever_levels:
                        self.ever_levels = cur
                        self.level_events.append({
                            "actions": self.stats["actions"],
                            "from": self.level_idx, "to": cur,
                            "goal": self._goal_snapshot,
                        })
                    self.on_level_complete(obs)
                self.level_idx = cur
                self.on_level_start(fresh=(self.stats["resets"] == 1 and
                                           self.stats["actions"] == 0))
                self.consec_fails = 0
            if not obs.frame:
                obs = env.step(GameAction.RESET)
                self.stats["resets"] += 1
                continue

            objs, sig, _bg = parse_frame(obs.frame[-1])
            self.since_progress += 1
            # MDL graft: register the observed state in the coarse-space
            # archive (predictions are judged here, not in 64px space).
            f32b = coarsen_frame(obs.frame[-1])
            self.mdl.note_coarse(coarse_sig32(f32b))
            if self.enable_goals:
                self.goals.observe(objs, obs.frame[-1])
            # Learned success signal: log a feature vector every 4th step
            # for the current attempt (labels come at outcome time).
            # Read-only w.r.t. goals (cached analysis), no RNG: behavior-
            # neutral, toy-gate safe.
            if ((not self.success_off or self.success_dump)
                    and self.enable_goals
                    and self.stats["actions"] % 4 == 0):
                self._log_success_state(objs)

            # Level-start bookkeeping once we have a trustworthy frame.
            # NOTE: the frame returned by the level-winning step still shows
            # the OLD level (sprites load on the next action), so after a
            # mid-game level change we skip one frame before trusting it.
            if self._start_mode is not None:
                if self._start_wait > 0:
                    self._start_wait -= 1
                    legal = self.legal_keys(obs, objs)
                    key = (self.known_good if self.known_good in legal
                           else (legal[0] if legal else None))
                    if key is None:
                        obs = env.step(GameAction.RESET)
                        self.stats["resets"] += 1
                        continue
                    self._goal_snapshot = self._snapshot_goal_state(False)
                    obs = self.do_step(key, objs, sig, f32b, obs.frame[-1],
                                       learn=False)
                    continue
                self._start_mode = None

            legal = self.legal_keys(obs, objs)
            is_new = self.archive.visit(sig, legal)
            if self.archive.cells[sig]["traj"] is None:
                self.archive.set_traj(sig, list(self.traj))
            if is_new:
                self.since_progress = 0
                if self.verbose:
                    print("  new cell #%d" % len(self.archive))

            # per-level attempt cap (keeps a doomed level from eating budget)
            if self.level_total[self.level_idx] > self.level_cap():
                self.stats["level_cap_resets"] += 1
                obs = env.step(GameAction.RESET)
                self.stats["resets"] += 1
                self.traj = []
                self.traj_deltas = []
                self._succ_log = []  # success-signal attempt log
                self.goal_plan = []
                self.goals.abort()
                self._start_mode = "reset"
                self._start_wait = 0
                continue

            key = self.maybe_goal_key(objs, legal)
            from_goal = key is not None
            if not from_goal:
                key = self.select(obs, objs, sig, legal, f32b)
            if key is None:
                # policy is stuck: Go-Explore return to a frontier cell
                self._goal_snapshot = self._snapshot_goal_state(False)
                obs = self.frontier_return()
                continue
            self._goal_snapshot = self._snapshot_goal_state(from_goal)
            if from_goal and self.first_goal_action is None:
                self.first_goal_action = self.stats["actions"]
            obs = self.do_step(key, objs, sig, f32b, obs.frame[-1],
                               learn=True)
            if from_goal:
                self.after_goal_step(obs)
            if self.surprise:
                # world-model violation: the level is probably in a hidden
                # doomed state -> cut the attempt short (RESET is free) and
                # tabu the trajectory so we don't repeat it.
                self.surprise = False
                self.stats["surprises"] += 1
                if from_goal:
                    # an active goal test is inconclusive after a surprise
                    self.goals.refute("surprise")
                    self.goal_plan = []
                if self.traj:
                    self.failed_trajs.add(tuple(self.traj))
                    while len(self.failed_trajs) > MAX_TABU:
                        self.failed_trajs.pop()
                if self.verbose:
                    print("  SURPRISE -> early reset (tabu=%d)" %
                          len(self.failed_trajs))
                obs = env.step(GameAction.RESET)
                self.stats["resets"] += 1
                self.traj = []
                self.traj_deltas = []
                self._succ_log = []  # success-signal attempt log
                self._start_mode = "reset"
                self._start_wait = 0
                continue

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def do_step(self, key, objs, sig, f32b, frame64=None, learn=True):
        act, data = self.resolve(key, objs)
        click64 = ((int(data["x"]), int(data["y"]))
                   if data else None)
        # prediction BEFORE the step, for surprise detection (world-model
        # violation -> the level is probably in a hidden doomed state).
        # Statistics prediction with the original n>=3 gate. (The MDL
        # program's native space is coarse/grid, not 64px signatures, so
        # it cannot supply the predicted added_colors this needs.)
        pred_added = None
        if learn:
            st = self.model.stats.get(key)
            if st and st["n"] >= 3:
                _psig, pred_added = self.model.predict(key, objs)
        self.surprise = False
        obs2 = self.env.step(act, data=data)
        self.stats["actions"] += 1
        self.level_actions[self.level_idx] += 1
        self.level_total[self.level_idx] += 1
        self.archive.mark_tried(sig, key)
        self.traj.append(key)
        self.seen_keys.add(key)
        if obs2.state is GameState.GAME_OVER:
            self.model._st(key)["fatal"] = \
                self.model._st(key).get("fatal", 0) + 1
            self.mdl.note_gameover(key)
            return obs2
        if (obs2.state is GameState.NOT_FINISHED
                and obs2.levels_completed == self.level_idx and obs2.frame):
            objs2, _sig2, _bg = parse_frame(obs2.frame[-1])
            if learn:
                added, removed = self.model.observe(key, objs, objs2)
                if frame64 is not None and f32b is not None:
                    f64a = obs2.frame[-1]
                    self.mdl.observe(key, frame64, f64a, click64)
                    if self.mdl.eval is not None:
                        self.mdl.record_eval(
                            key, f32b, coarsen_frame(f64a), objs, objs2,
                            click64, self.model.predict_objects)
            else:
                # blind step: don't train the model on a possibly-stale
                # transition, but still record the delta for valence/tabu.
                added, removed = TransitionModel.delta(objs, objs2)
            self.traj_deltas.append((key, added, removed))
            if (learn and pred_added and added
                    and not (pred_added & added)
                    and (any(c in self.good_colors for c in pred_added)
                         or any(c in self.bad_colors for c in added))):
                self.surprise = True
        return obs2

    def level_cap(self):
        i = self.level_idx
        b = self.baselines[i] if 0 <= i < len(self.baselines) else 60
        return max(1200, 4 * b)

    # ------------------------------------------------------------------
    # level lifecycle
    # ------------------------------------------------------------------

    def on_level_start(self, fresh):
        self.traj = []
        self.traj_deltas = []
        self._succ_log = []  # success-signal attempt log
        self.failed_trajs = set()
        self.consec_fails = 0
        self.goal_plan = []
        self.goals.new_level()
        self.stats["levels_seen"] += 1
        if fresh:
            self._start_mode = "reset"
            self._start_wait = 0
        else:
            self._start_mode = "levelchange"
            self._start_wait = 1  # skip the stale old-level frame

    def _refresh_valence(self):
        self.good_colors = {c for c, n in self.color_win.items() if n > 0}
        self.bad_colors = {c for c, n in self.color_loss.items()
                           if n > 0 and self.color_win.get(c, 0) == 0}

    def _net_added(self):
        """Colors whose object-count strictly increased during this level
        attempt (from the recorded per-step deltas). Net counting means
        pre-existing colors only register if the level actually added more
        of them -- this captures e.g. "bad sprites kept appearing" even
        when that color was already present at level start."""
        net = Counter()
        for _k, added, removed in self.traj_deltas:
            for c in added:
                net[c] += 1
            for c in removed:
                net[c] -= 1
        return {c for c, n in net.items() if n > 0}

    def on_game_over(self):
        self.color_loss.update(self._net_added())
        self._refresh_valence()
        self.goals.on_game_over()  # refutes an in-flight goal test
        self.goals.abort_pursuit()  # a pursuit step that died: inconclusive
        # Learned success signal: terminal states are what failure looks
        # like (y=0). Refit is ms-scale; game-overs are the abundant
        # negative class the sparse completion signal lacks.
        if self._succ_log and (not self.success_off or self.success_dump):
            term = self._succ_log[-2:]
            if not self.success_off:
                self.success.add(term, [0.0] * len(term))
                self.success.refit()
            if self.success_dump:
                self._succ_dump_X.extend(term)
                self._succ_dump_y.extend([0.0] * len(term))
        if self.traj:
            self.failed_trajs.add(tuple(self.traj))
            while len(self.failed_trajs) > MAX_TABU:
                self.failed_trajs.pop()
        self.stats["game_overs"] += 1
        if self.verbose:
            print("  GAME_OVER lvl=%d (tabu=%d)" %
                  (self.level_idx, len(self.failed_trajs)))

    def on_level_complete(self, obs):
        lvl = self.level_idx
        self.color_win.update(self._net_added())
        self._refresh_valence()
        self.have_win = True
        had_test = self.goals.testing is not None
        self.goals.confirm(lvl)  # confirms an in-flight goal test, if any
        if (not had_test and self.enable_goals and obs is not None
                and obs.frame):
            # Hindsight credit: the frame that won the level still shows
            # the OLD level (sprites load on the next action), so a pursued
            # hypothesis whose predicate holds in it likely describes the
            # goal -- credit it even though no plan test was in flight.
            wobjs, _, _ = parse_frame(obs.frame[-1])
            h = self.goals.hindsight(wobjs, lvl, self.stats["actions"])
            if h is not None and self.verbose:
                print("  GOAL-HINDSIGHT confirmed %s" % (h["id"][0],))
        self.goal_plan = []
        self.solutions[lvl] = list(self.traj)
        if self.traj:
            self.known_good = Counter(self.traj).most_common(1)[0][0]
        # Learned success signal: the winning attempt's logged states get
        # progress labels y=i/(n-1); refit so level-0 data informs level 1.
        if len(self._succ_log) >= 2 and \
                (not self.success_off or self.success_dump):
            n = len(self._succ_log)
            ys = [i / (n - 1) for i in range(n)]
            if not self.success_off:
                self.success.add(self._succ_log, ys)
                self.success.refit()
            if self.success_dump:
                self._succ_dump_X.extend(self._succ_log)
                self._succ_dump_y.extend(ys)
        if self.verbose:
            print("  LEVEL %d COMPLETE in %d actions (known_good=%s)" %
                  (lvl, self.level_actions[lvl], self.known_good))

    # ------------------------------------------------------------------
    # action selection
    # ------------------------------------------------------------------

    def select(self, obs, objs, sig, legal, f32b):
        """Returns an action-key, or None to trigger a frontier return."""
        if not legal:
            return None
        tried = self.archive.tried(sig)
        if self.rng.random() < 0.04:
            return self.rng.choice(legal)
        scored = [(self.score(key, objs, sig, tried, legal, f32b), key)
                  for key in legal]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_s, best_k = scored[0]
        if best_s < STUCK_SCORE and self.stats["returns"] < MAX_RETURNS:
            if self.archive.frontier() is not None:
                plan = self.bfs_plan(objs, sig, legal)
                if plan:
                    self.stats["plans_used"] += 1
                    return plan[0]
                return None
        return best_k

    def _tabu_penalty(self, key):
        """Penalize keys that would continue a known-failed trajectory."""
        t = tuple(self.traj) + (key,)
        for ft in self.failed_trajs:
            if len(ft) >= len(t) and ft[:len(t)] == t:
                return 2.0
        return 0.0

    def score(self, key, objs, sig, tried, legal, f32b=None):
        """Curiosity + valence scoring, two prediction levels.

        The predicted-outcome source depends on program maturity: keys
        whose MDL program is mature use A's program induction in its
        NATIVE coarse space (predicted coarse novelty in coarse_sig32,
        grid-exact no-op detection); keys without a mature program use
        B's statistics table exactly as before (fine-signature novelty,
        stats no-op rate). The valence channel (added colors) always
        comes from B's statistics, which natively capture stochastic
        appearance effects a deterministic program point-estimate misses.
        MDL falsification adds a bounded exploration bonus. No 64px
        signature is ever synthesized from a subsampled grid prediction
        (that bridge loses thin structure; see RESULTS_W2_MDL.md).
        """
        s = self.rng.random() * 0.05
        s -= self._tabu_penalty(key)
        st = self.model.stats.get(key)
        # prediction-error curiosity from the MDL graft: a recently
        # falsified program means "something here I don't understand yet".
        s += self.mdl.curiosity_bonus(key)
        if st is None or st["n"] == 0:
            # globally untried: pure curiosity (grid clicks deprioritized --
            # object clicks first, empty-space probes as fallback)
            return s + (0.8 if key[0] == "g" else 1.2)
        if st.get("fatal", 0) > 0:
            s -= 2.0  # observed to cause immediate GAME_OVER
        if key not in tried:
            s += 2.0  # untried from this state
        # valence channel: statistics only (added colors); predicted fine
        # signature is the original statistics fallback for keys without
        # a mature MDL program.
        psig, added = self.model.predict(key, objs)
        if f32b is not None and self.mdl.mature(key):
            # MDL-primary level: judge the program's prediction in coarse
            # space, where predicted vs observed are apples-to-apples.
            click64 = self._click64(key, objs)
            pcs = self.mdl.coarse_psig(key, f32b, click64)
            if pcs is not None:
                v = self.mdl.coarse_visits.get(pcs, 0)
                if v == 0:
                    s += 1.5  # predicted to reach a novel state
                else:
                    s += 0.6 / (1 + v)
            # grid-exact no-op detection: sharper than the statistics'
            # signature-based no-op rate (a program that predicts <=2
            # changed cells is a true no-op even when the fine signature
            # coincidentally wobbles). Disabled by default: a program
            # can be blind to sub-32x32 effects, so a "coarse no-op"
            # may still matter in fine space (toy-gate regression).
            if (os.environ.get("MDL_NOOP_PENALTY") == "1"
                    and self.mdl.predicts_noop(key, f32b, click64)):
                s -= 1.0
        else:
            # statistics level (original B behavior).
            if psig is not None:
                cell = self.archive.cells.get(psig)
                if cell is None:
                    s += 1.5  # predicted to reach a novel state
                else:
                    s += 0.6 / (1 + cell["visits"])
            if st["noop"] / st["n"] > 0.7:
                s -= 1.0  # usually a no-op
        if self.have_win:
            for c in added:
                if c in self.good_colors:
                    s += 1.0
                elif c in self.bad_colors:
                    s -= 3.0
        if key == self.known_good:
            s += 1.0 * (0.5 ** self.consec_fails)
        return s

    # ------------------------------------------------------------------
    # Go-Explore return + model-based planning
    # ------------------------------------------------------------------

    def frontier_return(self):
        """RESET (free) and replay the stored trajectory to a frontier cell."""
        self.stats["returns"] += 1
        self.goal_plan = []
        self.goals.abort()
        env = self.env
        fsig = self.archive.frontier()
        obs = env.step(GameAction.RESET)
        self.stats["resets"] += 1
        self.traj = []
        self.traj_deltas = []
        self._succ_log = []  # success-signal attempt log
        self._start_mode = None  # start colors already known for this level
        if fsig is None:
            return obs
        ftraj = list(self.archive.cells[fsig]["traj"] or [])
        for key in ftraj[:40]:
            if (obs.state is not GameState.NOT_FINISHED
                    or obs.levels_completed != self.level_idx
                    or not obs.frame):
                break
            objs, sig, _bg = parse_frame(obs.frame[-1])
            if key not in self.legal_keys(obs, objs):
                break
            obs = self.do_step(key, objs, sig, coarsen_frame(obs.frame[-1]),
                               obs.frame[-1], learn=True)
            if obs.state is not GameState.NOT_FINISHED:
                break
        if obs.state is GameState.NOT_FINISHED and obs.frame:
            _o, sig2, _b = parse_frame(obs.frame[-1])
            if sig2 != fsig:
                self.stats["return_mismatch"] += 1
        if self.verbose:
            print("  return->frontier trajlen=%d" % len(ftraj))
        return obs

    def bfs_plan(self, objs, sig, legal):
        """Bounded BFS over the learned object model toward frontier sigs."""
        targets = set(self.archive.frontier_sigs())
        if not targets:
            return None
        keys = [k for k in legal
                if self.model.stats.get(k, {}).get("n", 0) > 0
                and self.model.stats[k].get("fatal", 0) == 0][:10]
        if not keys:
            return None
        dq = deque([(objs, [])])
        seen = {sig}
        while dq:
            o, plan = dq.popleft()
            if len(plan) >= PLAN_DEPTH:
                continue
            for k in keys:
                psig, _added = self.model.predict(k, o)
                if psig is None or psig in seen:
                    continue
                seen.add(psig)
                nplan = plan + [k]
                if psig in targets:
                    return nplan
                po, _a = self.model.predict_objects(k, o)
                dq.append((po, nplan))
        return None

    # ------------------------------------------------------------------
    # goal-hypothesis mode
    # ------------------------------------------------------------------

    def _snapshot_goal_state(self, from_goal):
        """Diagnostic: describe the goal state at the decision point.

        Saved before every step; a later level-completion event uses the
        snapshot taken at the decision that issued the winning action.
        Read-only -- no behavior change."""
        g = self.goals
        testing = g.testing[0] if g.testing else None
        ph = g.pursuit_hyp
        return {
            "from_goal": bool(from_goal),
            "test": testing,
            "pursuit": ph["id"][0] if ph else None,
            "test_mode": bool(self.goal_plan_is_test),
            "plan_len": len(self.goal_plan),
        }

    def _model_mature(self, legal):
        """Enough learned dynamics to make planning worthwhile."""
        n = sum(1 for k in legal
                if self.model.stats.get(k, {}).get("n", 0) >= MIN_MODEL_N)
        return n >= MIN_MODEL_KEYS

    def maybe_goal_key(self, objs, legal):
        """A planned key for the active goal test, or launch a new test.

        Returns None when curiosity should stay in charge.
        """
        if not self.enable_goals:
            return None
        if self.goal_plan:
            while self.goal_plan and self.goal_plan[0] not in legal:
                self.goal_plan.pop(0)
            if self.goal_plan:
                return self.goal_plan.pop(0)
            # plan exhausted (remaining keys went illegal) without the
            # level completing -> inconclusive, not a refutation
            if self.goal_plan_is_test and self.goals.testing:
                self.goals.note_plan_fail(self.goals.testing)
                self.goals.abort()
            else:
                self.goals.abort_pursuit()
            return None
        if self.stats["actions"] < self.goal_eligible_at:
            return self._maybe_interleave(objs, legal)
        if len(self.goals.frames) < 6:
            return None
        stall = self.since_progress >= GOAL_K
        periodic = self.stats["actions"] - self.last_goal_test >= GOAL_M
        if not (stall or periodic):
            return self._maybe_interleave(objs, legal)
        self.goal_eligible_at = self.stats["actions"] + GOAL_COOLDOWN
        if not self._model_mature(legal):
            return self._maybe_interleave(objs, legal)
        # Try up to N_HYPS_PER_EVAL hypotheses per evaluation (was: only
        # the top one): unplannable hypotheses are skipped via plan-fails
        # instead of burning the whole evaluation on them.
        # Round 3: the stochastic beam climbs the learned success signal
        # V(s) (score_fn) rather than hand-written template progress --
        # the templates provide falsifiable test structure, V decides
        # where to go.
        vfn = None if self.success_off else self._success_value_fn(objs)
        for _ in range(N_HYPS_PER_EVAL):
            hyp = self.goals.top_hypothesis(objs, model=self.model,
                                            require_manipulable=True)
            if hyp is None:
                break
            # Deterministic plan first; else stochastic beam over sampled
            # outcomes. A *full* plan (predicate achieved, validated)
            # launches as a formal test; a *partial* plan with enough
            # predicted progress launches as a sustained pursuit (no
            # refutation -- composing multi-step sequences that single
            # best-effort steps cannot).
            plan = self.goals.plan_for(hyp, objs, self.model, legal)
            achieved = plan is not None
            if plan is None:
                plan, achieved, gain = self.goals.plan_beam(
                    hyp, objs, self.model, legal, self.goal_rng,
                    score_fn=vfn)
            else:
                gain = 1.0
            if not plan or (not achieved and gain < PURSUIT_GAIN_MIN):
                self.goals.note_plan_fail(hyp["id"])
                continue
            self.last_goal_test = self.stats["actions"]
            if achieved:
                self.goals.begin_test(hyp)
                self.goal_plan_is_test = True
            else:
                self.goals.begin_pursuit(hyp, self.stats["actions"])
                self.goal_plan_is_test = False
                plan = plan[:PURSUIT_PLAN_CAP]
            self.goal_plan_is_value = False
            self.goal_plan = list(plan)
            if self.verbose:
                print("  GOAL-%s %s params=%s plan=%s" %
                      ("TEST" if achieved else "PURSUIT",
                       hyp["id"][0], hyp["params"], plan))
            return self.goal_plan.pop(0)
        # Round 3 fallback: template-free value pursuit. When no hypothesis
        # yields a plan (no detected visual target, templates exhausted),
        # beam-search purely for high-V states. No predicate, no test
        # semantics -- always a pursuit (never refuted). Stall/periodic
        # gating keeps this a no-op on the toys (they solve before either
        # trigger fires), so the toy gate stays bit-identical.
        if vfn is not None:
            plan, gain = self.goals.plan_value(
                objs, self.model, legal, self.goal_rng, vfn,
                node_budget=2000)
            if plan and gain >= PURSUIT_GAIN_MIN:
                self.last_goal_test = self.stats["actions"]
                self.goal_plan = plan[:PURSUIT_PLAN_CAP]
                self.goal_plan_is_test = False
                self.goal_plan_is_value = True
                if self.verbose:
                    print("  GOAL-VALUE-PURSUIT plan=%s gain=%.2f" %
                          (plan, gain))
                return self.goal_plan.pop(0)
        return self._maybe_interleave(objs, legal)

    def _maybe_interleave(self, objs, legal):
        """Goal-directed from the first frames (direction d).

        Every GOAL_EVERY decisions, run a short beam search toward the top
        *region-grounded, manipulable* hypothesis and execute the resulting
        plan as a sustained pursuit: a validated full plan becomes a formal
        test, otherwise a partial plan with enough predicted progress is
        pursued for up to PURSUIT_PLAN_CAP steps (composing the multi-step
        sequences single best-effort steps cannot), else one best-effort
        step. Region-grounding (a detected outline/static target to aim
        at) keeps this a no-op on games where no visual target was ever
        detected, so toy behavior stays bit-identical.
        """
        if self.stats["actions"] - self._last_interleave < GOAL_EVERY:
            return None
        if len(self.goals.frames) < 6:
            return None
        if not any(self.model.stats.get(k, {}).get("n", 0) >= MIN_MODEL_N
                   for k in legal):
            return None
        self._last_interleave = self.stats["actions"]
        hyp = self.goals.top_hypothesis(objs, model=self.model,
                                        region_only=True,
                                        require_manipulable=True)
        if hyp is None:
            return None
        # Round 3: the interleave beam climbs the learned V(s) instead of
        # hand-written template progress. Toy-safe: this point is only
        # reached when a region-grounded hypothesis exists, which never
        # happens on the toys.
        vfn = None if self.success_off else self._success_value_fn(objs)
        plan, achieved, gain = self.goals.plan_beam(
            hyp, objs, self.model, legal, self.goal_rng, node_budget=2000,
            score_fn=vfn)
        if plan and (achieved or gain >= PURSUIT_GAIN_MIN):
            if achieved:
                self.goals.begin_test(hyp)
                self.goal_plan_is_test = True
            else:
                self.goals.begin_pursuit(hyp, self.stats["actions"])
                self.goal_plan_is_test = False
                plan = plan[:PURSUIT_PLAN_CAP]
            self.goal_plan_is_value = False
            self.goal_plan = list(plan)
            if self.verbose:
                print("  GOAL-%s %s params=%s plan=%s" %
                      ("TEST" if achieved else "PURSUIT",
                       hyp["id"][0], hyp["params"], plan))
            return self.goal_plan.pop(0)
        key = self.goals.best_effort_key(hyp, objs, self.model, legal,
                                         self.goal_rng)
        if key is None:
            return None
        self.goals.begin_pursuit(hyp, self.stats["actions"])
        self.goal_plan_is_test = False
        self.goal_plan_is_value = False
        if self.verbose:
            print("  GOAL-PURSUIT %s params=%s key=%s" %
                  (hyp["id"][0], hyp["params"], key))
        return key

    def after_goal_step(self, obs):
        """Score the active goal test / pursuit after one goal key.

        Verdict rule: a hypothesis is refuted only if its predicate was
        actually achieved without completing the level. If the step failed
        to achieve the predicate, a plan test is inconclusive (bad plan,
        not necessarily a bad hypothesis) and the hypothesis stays
        untried; a pursuit (single-step or sustained partial plan) is
        likewise inconclusive -- the plan was never claimed to achieve
        the predicate, so ending it must not refute the hypothesis.
        """
        g = self.goals
        if g.testing is None and g.pursuit_hyp is None:
            return
        if obs.state is GameState.GAME_OVER:
            return  # loop-top on_game_over() refutes tests / aborts pursuits
        if obs.levels_completed != self.level_idx:
            return  # completion is confirmed at the top of the loop
        if self.goal_plan:
            return  # plan test / pursuit still in progress
        is_test = self.goal_plan_is_test and g.testing is not None
        hyp = g.testing_hyp if is_test else g.pursuit_hyp
        if hyp is not None and obs.frame:
            objs2, _, _ = parse_frame(obs.frame[-1])
            if g.satisfied(hyp, objs2):
                if is_test:
                    g.refute("predicate-true-no-completion")
                # pursuit (single-step or partial plan): predicate achieved
                # but no completion. Don't refute -- the already-true
                # check will handle it on the next evaluation.
                else:
                    g.abort_pursuit()
                return
        if is_test:
            g.note_plan_fail(g.testing)
            g.abort()
        else:
            g.abort_pursuit()  # inconclusive: hypothesis stays untried

    # ------------------------------------------------------------------
    # learned success signal (round 3)
    # ------------------------------------------------------------------

    def _succ_ctx(self):
        """Trajectory/scene context for success features (per decision).

        Static structure comes from the cached goal-module analysis; the
        two trajectory features are constant during a planning call, so
        imagined states reuse this context.
        """
        an = self.goals._analyze()
        return {
            "static_anchors": an["static_anchors"],
            "regions": an["regions"],
            "movers": an["movers"],
            "actions_frac": (self.level_total[self.level_idx]
                             / max(1, self.level_cap())),
            "arch_frac": math.log1p(len(self.archive)) / math.log1p(4096.0),
        }

    def _log_success_state(self, objs):
        """Append one feature vector to the current attempt's log.

        Read-only w.r.t. the goal module (cached analysis), no RNG --
        behavior-neutral by construction.
        """
        try:
            self._succ_log.append(
                featurize(objs, self.goals, ctx=self._succ_ctx()))
        except Exception:
            self.stats["succ_feat_fail"] = \
                self.stats.get("succ_feat_fail", 0) + 1

    def _success_value_fn(self, objs):
        """V(s) scorer for *imagined* states, bound to the current scene.

        Hypotheses and init-counts are bound once per planning call and
        reused for every imagined state; returns None when the learned
        signal is disabled.
        """
        if self.success_off:
            return None
        if not self.success.has_signal():
            # No learned signal yet (zero prior, no online data): V(s)=0
            # everywhere would make the beam uninformed. Fall back to
            # template-progress scoring until the model learns something.
            return None
        g = self.goals
        hyps = g.hypotheses(objs)[:6]
        inits = {id(h): g._init_counts(h, objs) for h in hyps}
        ctx = self._succ_ctx()
        success = self.success

        def vf(po):
            return success.value(featurize(po, g, hyps, inits, ctx))

        return vf

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def finish(self, reason, t0):
        wall = time.time() - t0
        if self.success_dump and self._succ_dump_X:
            try:
                import numpy as _np
                os.makedirs(self.success_dump, exist_ok=True)
                tag = os.environ.get("SUCCESS_DUMP_TAG", "run")
                _np.savez(os.path.join(self.success_dump, tag + ".npz"),
                          X=_np.array(self._succ_dump_X),
                          y=_np.array(self._succ_dump_y))
            except Exception as e:
                print("success dump failed: %s" % e)
        return {
            "stop": reason,
            "wall_s": wall,
            "actions": self.stats["actions"],
            "resets": self.stats["resets"],
            "game_overs": self.stats["game_overs"],
            "levels_completed": self.last_levels,
            "levels_ever": self.ever_levels,
            "level_events": self.level_events,
            "first_goal_action": self.first_goal_action,
            "level_actions": dict(self.level_actions),
            "solutions": {k: len(v) for k, v in self.solutions.items()},
            "archive_size": len(self.archive),
            "model_keys": len(self.model.stats),
            "known_good": self.known_good,
            "returns": self.stats["returns"],
            "return_mismatch": self.stats["return_mismatch"],
            "plans_used": self.stats["plans_used"],
            "surprises": self.stats["surprises"],
            "level_cap_resets": self.stats["level_cap_resets"],
            "goal": self.goals.summary(),
            "mdl": self.mdl.summary(),
            "mdl_eval": self.mdl.eval_summary(),
            "success": self.success.summary(),
        }
