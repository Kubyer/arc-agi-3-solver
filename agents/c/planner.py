"""Planning as intervention on the learned SCM.

Goal discovery: when levels_completed increments we record the winning
transition's signed variable changes and maintain an exponential moving
average = the "win direction" per variable, with weights proportional to how
strongly each variable moved during wins. The goal is a *direction* in
variable space (transferable across levels/contexts), not a fixed state.

Control: greedy 1-step causal controller --
    score(I) = sum_V w_V * sign(win_d_V) * E[delta_V | do(I)] * conf(I->V)
               - DEATH_PENALTY * P(died | do(I))
               + WIN_BONUS     * P(win  | do(I))
               + UCB exploration bonus
i.e. simulate each candidate intervention on the SCM and pick the one whose
predicted effect best reproduces the win direction.

Fallback: if the raw state stalls, beam-search the SCM forward (most-likely
transitions) for a short sequence maximizing win-direction progress.
"""

import math
import random

from variables import VAR_NAMES, BIN_MAX

DEATH_PENALTY = 6.0
WIN_BONUS = 4.0
DOOM_PENALTY = 5.0


class Planner:
    def __init__(self, model, seed=0):
        self.model = model
        self.rng = random.Random(seed)
        self.win_d = {}   # V -> EMA of signed bin-change in winning transitions
        self.win_w = {}   # V -> normalized importance weight
        self.n_wins = 0

    # ---------------------------------------------------------- goal learning
    def on_win(self, ds):
        """Incorporate one winning transition's signed changes."""
        self.n_wins += 1
        a = 0.45
        for V, d in ds.items():
            self.win_d[V] = (1 - a) * self.win_d.get(V, 0.0) + a * d
        tot = sum(abs(v) for v in self.win_d.values()) + 1e-9
        self.win_w = {V: abs(v) / tot for V, v in self.win_d.items()}

    def has_goal(self):
        return self.n_wins > 0 and any(abs(v) > 0.05
                                       for v in self.win_d.values())

    # ---------------------------------------------------------------- scoring
    def _progress(self, I, edges):
        s = 0.0
        for V in self.model.active_vars():
            e = edges.get((I, V))
            if e is None:
                continue
            wd = self.win_d.get(V, 0.0)
            if abs(wd) < 0.05:
                continue
            s += self.win_w[V] * (1 if wd > 0 else -1) * e.mu * e.conf
        return s

    def _doom_of(self, I, state_bins, ctx):
        m = self.model
        post = m.predict(I, state_bins, ctx)
        return m.doom_prob(post)

    def greedy_scores(self, state_bins, ctx, labels):
        """Return dict label -> score for candidate interventions."""
        m = self.model
        edges = m.action_edges()
        N = m.n_total
        out = {}
        for I in labels:
            nI = m.n_I.get(I, 0)
            doom = self._doom_of(I, state_bins, ctx)
            if self.has_goal():
                s = self._progress(I, edges)
                s -= DEATH_PENALTY * m.p_outcome(I, "died")
                s += WIN_BONUS * m.p_outcome(I, "win")
                s += 0.6 * math.sqrt(math.log(1 + N) / (1 + nI))
                # boredom: penalize interventions known to do nothing
                if nI > 4:
                    mx = max((abs(e.mu) for (i, v), e in edges.items()
                              if i == I), default=0.0)
                    if mx < 0.05:
                        s -= 0.4
                # near-ban on lethal interventions (unless nothing else)
                if m.p_outcome(I, "died") > 0.4 and nI >= 3:
                    s *= 0.05
            else:
                # no goal yet: active causal learning (curiosity).
                # Prefer interventions that *do* things: dense pixel-change
                # signal (catches rearrangements bins miss) + binned changes.
                s = 2.0 * math.sqrt(math.log(1 + N) / (1 + nI))
                s += 0.4 * m.expected_changes(I)
                s += 0.6 * min(1.0, m.mean_px_change(I) / 150.0)
                s -= DEATH_PENALTY * 0.5 * m.p_outcome(I, "died")
            s -= DOOM_PENALTY * doom
            out[I] = s + self.rng.random() * 1e-9
        return out

    def safe_candidates(self, state_bins, ctx, labels):
        """Candidates that are not known-lethal and not predicted doomed."""
        m = self.model
        ok = [I for I in labels
              if not (m.p_outcome(I, "died") > 0.4 and m.n_I.get(I, 0) >= 3)
              and self._doom_of(I, state_bins, ctx) < 0.6]
        return ok or list(labels)

    # ------------------------------------------------------------ beam search
    def _reward(self, bins, start):
        r = 0.0
        for V in self.model.active_vars():
            wd = self.win_d.get(V, 0.0)
            if abs(wd) < 0.05:
                continue
            rng = max(1, BIN_MAX[V])
            r += self.win_w[V] * (1 if wd > 0 else -1) * (bins[V] - start[V]) / rng
        return r

    def beam_search(self, state_bins, ctx, labels, depth=5, beam=48):
        """Search intervention sequences on the SCM; return first action."""
        m = self.model
        if not labels:
            return None
        # beams: (reward, path, state)
        beams = [(self._reward(state_bins, state_bins), [], state_bins)]
        seen = {tuple(sorted(state_bins.items()))}
        best = beams[0]
        for _ in range(depth):
            nxt = []
            for r, path, st in beams:
                for I in labels:
                    ns = m.predict(I, st, ctx)
                    key = tuple(sorted(ns.items()))
                    nr = self._reward(ns, state_bins)
                    nr -= DEATH_PENALTY * 0.3 * m.p_outcome(I, "died")
                    if key not in seen:
                        nr += 0.05  # novelty bonus
                    cand = (nr, path + [I], ns)
                    nxt.append(cand)
                    if nr > best[0]:
                        best = cand
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[0])
            beams = nxt[:beam]
            for _, _, st in beams:
                seen.add(tuple(sorted(st.items())))
        if best[1] and best[0] > self._reward(state_bins, state_bins) + 1e-6:
            return best[1][0]
        return None
