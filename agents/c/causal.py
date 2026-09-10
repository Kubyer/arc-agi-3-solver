"""Interventional causal discovery for the SCM world model.

Every environment step is an intervention do(I) on the game. We record, for
each (intervention, variable) pair, how the variable's binned value changes,
and test:

  action -> variable edge:  P(V changes | do(I)) >> P(V changes | do(other)),
                            with the effect *stable across contexts*
                            (invariance = the signature of a mechanism).

  variable -> variable edge: P(B changes | A changed) >> P(B changes | A same)
                            within the same transition (temporal order).

Context = (level index, color-presence bitmask). A mechanism is kept only if
its effect survives context shifts; spurious correlations do not.

Model selection: 3 candidate variable subsets (full / mid / min) are scored by
time-split one-step prediction accuracy, penalized by train/test gap
(invariance across time). The winner is used for planning.

All statistics are incremental (no batch recomputation), numpy-only.
"""

import math
from collections import Counter, defaultdict

from variables import VAR_NAMES, BIN_MAX


class _Edge:
    __slots__ = ("I", "V", "conf", "mu", "p_change")

    def __init__(self, I, V, conf, mu, p_change):
        self.I = I
        self.V = V
        self.conf = conf
        self.mu = mu            # mean signed bin-change under do(I)
        self.p_change = p_change

    def to_dict(self):
        return {"intervention": str(self.I), "variable": self.V,
                "confidence": round(self.conf, 3),
                "mean_delta": round(self.mu, 3),
                "p_change": round(self.p_change, 3)}


def _signed_delta(pre, post, var):
    d = post - pre
    return 1 if d > 0 else (-1 if d < 0 else 0)


class CausalModel:
    # candidate variable subsets for model selection by invariance
    CANDIDATES = {
        "full": VAR_NAMES,
        "mid": ["ncol", "fg", "d12"] + [f"{p}{k}" for k in range(1, 7)
                                        for p in ("cnt", "obj")],
        "min": ["ncol", "fg"] + [f"cnt{k}" for k in range(1, 7)],
    }

    def __init__(self, seed=0):
        self.n_total = 0
        self.n_I = Counter()          # samples per intervention
        # (I,V) -> [n, n_change, sum_signed_delta]
        self.st = defaultdict(lambda: [0, 0, 0.0])
        # (I,V,ctx) -> [n, n_change, sum_signed_delta]
        self.stc = defaultdict(lambda: [0, 0, 0.0])
        # (I,V) -> Counter(signed delta) for most-likely prediction
        self.mode = defaultdict(Counter)
        self.modec = defaultdict(Counter)
        # per-variable global change stats for the "other interventions" baseline
        self.chgV = defaultdict(lambda: [0, 0])   # V -> [n_change, n]
        # variable -> variable temporal stats:
        # (A,B) -> [n_Achg, n_Achg_Bchg, n_Asame, n_Asame_Bchg]
        self.vv = defaultdict(lambda: [0, 0, 0, 0])
        # outcomes per intervention: 'win' | 'died' | 'none'
        self.outc = defaultdict(Counter)
        # raw pixel-change magnitude per intervention (dense curiosity signal:
        # binned variables can miss rearrangements that conserve counts)
        self.chgpx = defaultdict(lambda: [0, 0])  # I -> [total_px, n]
        # doom statistics over STATES (credit assignment for delayed death):
        # seen[(V,bin)] = times observed; doomed[(V,bin)] = times the attempt
        # died within the horizon after such a state was visited.
        self.seen = defaultdict(int)
        self.doomed = defaultdict(int)
        # raw sample log (bounded) for time-split model selection
        self.samples = []
        self.active = "full"
        self._edges_cache = None
        self._vvedges_cache = None

    # ------------------------------------------------------------------ data
    def observe(self, pre_bins, post_bins, I, ctx, outcome, px_changed=0):
        """Record one interventional transition.

        Transitions with outcome='win' conflate the winning action with the
        level reload (post shows the fresh level), so they are kept for
        outcome probabilities but excluded from the (I,V) mechanism tables.
        """
        ds = {V: _signed_delta(pre_bins[V], post_bins[V], V)
              for V in VAR_NAMES}
        self.n_total += 1
        self.n_I[I] += 1
        self.outc[I][outcome] += 1
        cp = self.chgpx[I]
        cp[0] += px_changed
        cp[1] += 1
        if outcome != "win":
            for V in VAR_NAMES:
                d = ds[V]
                chg = 1 if d != 0 else 0
                s = self.st[(I, V)]
                s[0] += 1
                s[1] += chg
                s[2] += d
                sc = self.stc[(I, V, ctx)]
                sc[0] += 1
                sc[1] += chg
                sc[2] += d
                self.mode[(I, V)][d] += 1
                self.modec[(I, V, ctx)][d] += 1
                g = self.chgV[V]
                g[0] += chg
                g[1] += 1
            # var->var temporal co-change
            changed = {V for V in VAR_NAMES if ds[V] != 0}
            for A in VAR_NAMES:
                achg = 1 if A in changed else 0
                for B in VAR_NAMES:
                    if A == B:
                        continue
                    e = self.vv[(A, B)]
                    if achg:
                        e[0] += 1
                        e[1] += 1 if B in changed else 0
                    else:
                        e[2] += 1
                        e[3] += 1 if B in changed else 0
        if len(self.samples) < 6000:
            self.samples.append((I, ctx, dict(pre_bins), dict(post_bins),
                                 outcome))
        self._edges_cache = None
        self._vvedges_cache = None
        return ds

    # ----------------------------------------------------------------- edges
    def _invariance(self, I, V):
        """1 - CV of per-context change-probabilities (>=2 contexts)."""
        vals = []
        for (i, v, c), s in self.stc.items():
            if i == I and v == V and s[0] >= 4:
                vals.append(s[1] / s[0])
        if len(vals) < 2:
            return 0.7, 1  # single context: provisional
        m = sum(vals) / len(vals)
        if m <= 1e-9:
            return 0.0, len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)
        return max(0.0, 1.0 - math.sqrt(var) / m), len(vals)

    def action_edges(self, min_n=6, min_effect=0.25, min_conf=0.22):
        """Learn directed edges intervention -> variable (the causal graph)."""
        if self._edges_cache is not None:
            return self._edges_cache
        edges = {}
        for (I, V), s in self.st.items():
            n = s[0]
            if n < min_n:
                continue
            p_change = s[1] / n
            g = self.chgV[V]
            p_other = ((g[0] - s[1]) / (g[1] - n)) if g[1] > n else 0.0
            effect = p_change - p_other
            if effect < min_effect or p_change < 0.35:
                continue
            inv, nctx = self._invariance(I, V)
            conf = effect * min(1.0, n / 12.0) * (0.5 + 0.5 * inv)
            if conf >= min_conf:
                mu = s[2] / n
                edges[(I, V)] = _Edge(I, V, conf, mu, p_change)
        self._edges_cache = edges
        return edges

    def var_edges(self, min_n=15, min_effect=0.35):
        """Learn directed edges variable -> variable (temporal co-change)."""
        if self._vvedges_cache is not None:
            return self._vvedges_cache
        out = []
        for (A, B), e in self.vv.items():
            n1, n1b, n0, n0b = e
            if n1 < min_n or n0 < min_n:
                continue
            p1 = n1b / n1
            p0 = n0b / n0
            if p1 - p0 >= min_effect and p1 >= 0.4:
                out.append((A, B, round(p1 - p0, 3), round(p1, 3)))
        out.sort(key=lambda t: -t[2])
        self._vvedges_cache = out[:40]
        return self._vvedges_cache

    # -------------------------------------------------------------- prediction
    def predict_delta(self, I, V, ctx):
        """Most-likely signed bin change of V under do(I) (invariant part)."""
        mc = self.modec.get((I, V, ctx))
        if mc:
            tot = sum(mc.values())
            if tot >= 4:
                return mc.most_common(1)[0][0]
        m = self.mode.get((I, V))
        if m and sum(m.values()) >= 4:
            return m.most_common(1)[0][0]
        return 0

    def predict(self, I, pre_bins, ctx):
        """Forward-sample the SCM one step (most-likely outcome per var)."""
        edges = self.action_edges()
        post = dict(pre_bins)
        changed = set()
        for V in VAR_NAMES:
            e = edges.get((I, V))
            if e is None:
                continue
            d = self.predict_delta(I, V, ctx)
            if d != 0:
                post[V] = int(max(0, min(BIN_MAX[V], pre_bins[V] + d)))
                changed.add(V)
        # propagate through variable->variable edges (one hop)
        for A, B, eff, _ in self.var_edges():
            if A in changed and B not in changed:
                d = self.predict_delta(I, B, ctx)
                if d != 0:
                    post[B] = int(max(0, min(BIN_MAX[B], pre_bins[B] + d)))
        return post

    def p_outcome(self, I, outcome):
        c = self.outc.get(I)
        if not c:
            return 0.0
        tot = sum(c.values())
        return c[outcome] / tot if tot else 0.0

    def mean_px_change(self, I):
        cp = self.chgpx.get(I)
        if not cp or cp[1] == 0:
            return 0.0
        return cp[0] / cp[1]

    def expected_changes(self, I):
        """Mean #variables changed per do(I) (curiosity signal)."""
        n = self.n_I.get(I, 0)
        if n == 0:
            return 0.0
        tot = sum(s[1] for (i, v), s in self.st.items() if i == I)
        return tot / n

    def note_seen(self, bins):
        for V in VAR_NAMES:
            self.seen[(V, bins[V])] += 1

    def note_doomed(self, bins):
        for V in VAR_NAMES:
            self.doomed[(V, bins[V])] += 1

    def doom_prob(self, bins, min_seen=8):
        """Max over variables of P(attempt dooms | V=bin)."""
        best = 0.0
        for V in VAR_NAMES:
            s = self.seen.get((V, bins[V]), 0)
            if s >= min_seen:
                p = self.doomed.get((V, bins[V]), 0) / s
                if p > best:
                    best = p
        return best

    # -------------------------------------------------------- model selection
    def _split_accuracy(self, varset):
        """Time-split one-step prediction accuracy for a variable subset."""
        S = self.samples
        if len(S) < 60:
            return 0.0, 0.0
        k = int(len(S) * 0.7)
        train, test = S[:k], S[k:]
        # fit most-likely deltas on train (pooled = invariant mechanism)
        from collections import Counter as C
        tab = defaultdict(C)
        for (I, ctx, pre, post, o) in train:
            for V in varset:
                tab[(I, V)][_signed_delta(pre[V], post[V], V)] += 1
        pred = {(I, V): c.most_common(1)[0][0] for (I, V), c in tab.items()
                if sum(c.values()) >= 3}
        # accuracy on test
        hit = tot = 0
        acc_train_hit = acc_train_tot = 0
        for (I, ctx, pre, post, o) in test:
            for V in varset:
                d = pred.get((I, V), 0)
                p = pre[V] + d
                p = max(0, min(BIN_MAX[V], p))
                hit += 1 if p == post[V] else 0
                tot += 1
        for (I, ctx, pre, post, o) in train:
            for V in varset:
                d = pred.get((I, V), 0)
                p = pre[V] + d
                p = max(0, min(BIN_MAX[V], p))
                acc_train_hit += 1 if p == post[V] else 0
                acc_train_tot += 1
        ta = hit / tot if tot else 0.0
        tra = acc_train_hit / acc_train_tot if acc_train_tot else 0.0
        return ta, abs(tra - ta)

    def select_model(self):
        """Pick the candidate whose mechanism predicts best across time."""
        if self.n_total < 60:
            return self.active
        best, best_key = None, None
        for name, varset in self.CANDIDATES.items():
            ta, gap = self._split_accuracy(varset)
            key = (ta - 0.5 * gap, -len(varset))  # accuracy, invariance, parsimony
            if best_key is None or key > best_key:
                best_key, best = key, name
        if best and best != self.active:
            self.active = best
        return self.active

    def active_vars(self):
        return self.CANDIDATES[self.active]

    # ---------------------------------------------------------------- export
    def graph_dict(self):
        return {
            "active_model": self.active,
            "n_samples": self.n_total,
            "action_edges": [e.to_dict()
                             for e in self.action_edges().values()],
            "var_edges": [{"from": a, "to": b, "effect": e, "p": p}
                          for a, b, e, p in self.var_edges()],
            "outcome_probs": {str(I): {o: round(c[o] / sum(c.values()), 3)
                                       for o in c}
                              for I, c in self.outc.items()},
        }
