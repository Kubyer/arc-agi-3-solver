"""Object-level transition model.

For each action-key we accumulate simple statistics over observed
object-level deltas:
  - appear:    Counter of (color, size_bin) of objects present after but not before
  - disappear: Counter of (color, size_bin) of objects present before but not after
  - move:      per-color list of (dy, dx) centroid displacements of matched objects
  - noop:      fraction of steps where the signature did not change at all

Objects are matched greedily by color, then nearest centroid within a
distance threshold. The model supports two queries used by the agent:
  - predict(key, objs) -> (pred_sig, added_colors): coarse next-state forecast
    used for curiosity scoring (novelty of the predicted state) and for
    valence (which colors the action tends to introduce).
  - predict_objects(key, objs) -> (new_objs, added_colors): used by the
    BFS planner to expand imagined states.

Deliberately simple: per-action tabular statistics, no function
approximation. Context conditioning (e.g. "near border") is omitted for
now -- the statistics are pooled globally, which is enough when action
semantics are stationary within a game.
"""

import math
from collections import Counter, defaultdict

from .objects import signature

MATCH_DIST = 12.0  # max centroid distance (px) to match same-color objects


class TransitionModel:
    def __init__(self):
        self.stats = {}

    def _st(self, key):
        st = self.stats.get(key)
        if st is None:
            st = {"appear": Counter(), "disappear": Counter(),
                  "move": defaultdict(list), "appear_pos": defaultdict(list),
                  "n": 0, "noop": 0}
            self.stats[key] = st
        return st

    @staticmethod
    def _match(before, after):
        """Greedy same-color nearest matching.

        Returns (matches, unmatched_before, unmatched_after) where matches
        is a list of (b, a) pairs.
        """
        unmatched_a = list(after)
        matches, unmatched_b = [], []
        for b in before:
            best, best_d = None, MATCH_DIST
            for a in unmatched_a:
                if a["color"] != b["color"]:
                    continue
                d = math.hypot(a["cy"] - b["cy"], a["cx"] - b["cx"])
                if d < best_d:
                    best_d, best = d, a
            if best is None:
                unmatched_b.append(b)
            else:
                unmatched_a.remove(best)
                matches.append((b, best))
        return matches, unmatched_b, unmatched_a

    @staticmethod
    def _unmatched(before, after):
        """Returns (unmatched_before, unmatched_after) object lists."""
        _m, ub, ua = TransitionModel._match(before, after)
        return ub, ua

    @staticmethod
    def delta(before, after):
        """(added_colors, removed_colors) without recording anything."""
        ub, ua = TransitionModel._unmatched(before, after)
        return {a["color"] for a in ua}, {b["color"] for b in ub}

    def observe(self, key, before, after):
        """Record one (objs_before, action, objs_after) transition.

        Returns (added_colors, removed_colors): sets of colors of objects
        that appeared / disappeared in this transition.
        """
        matches, unmatched_b, unmatched_a = self._match(before, after)
        st = self._st(key)
        st["n"] += 1
        if signature(before) == signature(after):
            st["noop"] += 1
        for b, a in matches:
            st["move"][b["color"]].append((a["cy"] - b["cy"],
                                          a["cx"] - b["cx"]))
        for b in unmatched_b:
            st["disappear"][(b["color"], b["size_bin"])] += 1
        for a in unmatched_a:
            st["appear"][(a["color"], a["size_bin"])] += 1
            st["appear_pos"][(a["color"], a["size_bin"])].append(
                (a["cy"], a["cx"]))
        return {a["color"] for a in unmatched_a}, \
            {b["color"] for b in unmatched_b}

    # -- prediction ----------------------------------------------------

    def _mean_moves(self, st):
        mm = {}
        for c, ds in st["move"].items():
            if ds:
                mm[c] = (sum(d[0] for d in ds) / len(ds),
                         sum(d[1] for d in ds) / len(ds))
        return mm

    def _mean_moves_cached(self, key):
        """Mean displacement per color, cached per observation count.

        predict_objects is called in tight planning loops; recomputing
        means by summing the full displacement history on every call is
        O(history). The cache is keyed on st["n"], which increments on
        every observe(), so it is always fresh.
        """
        st = self.stats.get(key)
        if not st:
            return {}
        c = st.get("_mean_cache")
        if c is None or c[0] != st["n"]:
            mm = self._mean_moves(st)
            st["_mean_cache"] = (st["n"], mm)
            return mm
        return c[1]

    def predict_objects(self, key, objs):
        """Imagine the object list after applying key. Returns (objs, added)."""
        st = self.stats.get(key)
        if not st or st["n"] == 0:
            return [dict(o) for o in objs], set()
        n = st["n"]
        mm = self._mean_moves_cached(key)
        out = [dict(o) for o in objs]
        for o in out:
            d = mm.get(o["color"])
            if d:
                o["cy"] += d[0]
                o["cx"] += d[1]
        if st["disappear"]:
            (c, sb), cnt = st["disappear"].most_common(1)[0]
            if cnt / n > 0.5:
                cands = [o for o in out
                         if o["color"] == c and o["size_bin"] == sb]
                if cands:
                    out.remove(min(cands, key=lambda o: o["size"]))
        added = set()
        for (c, sb), cnt in st["appear"].most_common(3):
            if cnt / n > 0.35:
                pos = st["appear_pos"][(c, sb)]
                cy = sum(p[0] for p in pos) / len(pos)
                cx = sum(p[1] for p in pos) / len(pos)
                out.append({"color": c, "size": 4, "size_bin": sb,
                            "cy": cy, "cx": cx, "bbox": (0, 0, 0, 0)})
                added.add(c)
        return out, added

    def sample_objects(self, key, objs, rng):
        """Draw ONE plausible outcome from the empirical per-key outcome
        distribution (direction a: distributional transitions).

        - movement: each object draws a displacement from the recorded
          (dy, dx) samples of its color, with per-object move probability
          = fraction of samples with |disp| > 0.5 (stationary actions, e.g.
          clicks that only sometimes move a sprite, keep their mass at 0).
        - disappearance: each object vanishes w.p. disappear[(color, bin)]/n.
        - appearance: each recorded (color, bin) class materializes w.p.
          count/n at a uniformly sampled recorded position.

        Unlike predict_objects (means), repeated calls branch into the
        distinct outcomes the key actually produced, so Monte-Carlo
        rollouts / stochastic beam search can find plans the mean
        prediction smooths away (e.g. a click that moves a sprite by
        either 0 or 8px has mean 4px, which never occurs).
        """
        st = self.stats.get(key)
        if not st or st["n"] == 0:
            return [dict(o) for o in objs], set()
        n = st["n"]
        out = []
        # Cache the per-color displacement sample lists: st["move"][c]
        # grows with every observation, so looking it up once per color
        # (not once per object) keeps sampling O(objs), not
        # O(objs * history).
        move_cache = {}
        for o in objs:
            oo = dict(o)  # never alias the input list
            c = o["color"]
            ds = move_cache.get(c, False)
            if ds is False:
                ds = st["move"].get(c)
                move_cache[c] = ds
            if ds:
                # Draw from the empirical displacement distribution
                # directly -- it already contains (0,0) samples for
                # stationary observations, so no extra Bernoulli gate.
                dy, dx = ds[rng.randrange(len(ds))]
                oo["cy"] += dy
                oo["cx"] += dx
            p_gone = st["disappear"].get((o["color"], o["size_bin"]), 0) / n
            if rng.random() < min(1.0, p_gone):
                continue  # object disappears in this branch
            out.append(oo)
        added = set()
        for (c, sb), cnt in st["appear"].most_common(3):
            if rng.random() < min(1.0, cnt / n):
                pos = st["appear_pos"][(c, sb)]
                cy, cx = pos[rng.randrange(len(pos))]
                out.append({"color": c, "size": 4, "size_bin": sb,
                            "cy": float(cy), "cx": float(cx),
                            "bbox": (0, 0, 0, 0)})
                added.add(c)
        return out, added

    def predict(self, key, objs):
        """Return (predicted_sig, added_colors); (None, set()) if unknown."""
        st = self.stats.get(key)
        if not st or st["n"] == 0:
            return None, set()
        out, added = self.predict_objects(key, objs)
        return signature(out), added

    def effect_summary(self, key):
        """Colors this action tends to add/remove, with empirical probabilities."""
        st = self.stats.get(key)
        if not st or st["n"] == 0:
            return None
        n = st["n"]
        return {
            "added": {c: v / n for (c, _), v in st["appear"].items()},
            "removed": {c: v / n for (c, _), v in st["disappear"].items()},
            "noop_p": st["noop"] / n,
            "n": n,
        }
