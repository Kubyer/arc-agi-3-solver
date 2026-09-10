"""Learned success signal for the object-centric curiosity agent (round 3).

Replaces the hand-written goal-predicate templates as the *decision rule*
for what the agent pursues. A small L2-regularized linear model

    V(s) = clip(w . x(s), 0, 1)

predicts "progress toward level completion" from game-agnostic
object-relational features. The templates survive in two subordinate
roles: (a) as *features* -- each template's progress measure is 8 of the
24 inputs, so the model can learn which relational patterns actually
predict success; (b) as falsifiable *test structure* for formal goal
tests. The planner pursues high-V states directly.

Training signal (sparse but honest):
  - states along a *winning* attempt: label y = i/(n-1), the normalized
    position in the winning trajectory (progress already made);
  - *game-over* terminal states: y = 0 (what failure looks like).

Cold start: a prior trained offline on toy-game winning trajectories
(train_prior.py -> success_prior.json). Within a game, every completion
and game-over refits the model with prior-centered L2, so level-0 data
informs level-1 targets through the shared per-game model (transfer via
the game-agnostic feature space).

All features are computable from an object list alone -- no raw frame,
no archive rescan -- so V(s) can score *imagined* states inside the
planner's beam search. Every feature is scaled to ~[0,1] by construction,
so plain (unstandardized) ridge regression is well-conditioned; no
scikit-learn needed.

Env flags (read in agent.py):
  SUCCESS_OFF=1   disable the learned signal entirely (ablation baseline:
                  pure round-2 behavior, bit-identical).
  SUCCESS_DUMP=d  dump per-outcome (X, y) to d/*.npz at episode end for
                  offline prior training (implies logging, not influence).
"""

import json
import math
import os

import numpy as np

from .goals import TEMPLATE_INDEX

N_FEATURES = 24
PRIOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "success_prior.json")
# Prior-centered L2 strength: the prior counts as this many pseudo-samples.
PRIOR_STRENGTH = 30.0
MAX_DATA = 4000  # cap on per-game online samples (drop oldest)


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------

def _obj_features(objs):
    """Cheap object-inventory / organization features from an object list.

    Works on imagined (planner-hallucinated) object lists too: no frame,
    no goal-module analysis. All ~[0,1].
    """
    n = len(objs)
    if n == 0:
        return [0.0] * 9 + [0.0, 0.0]  # 11 numbers; caller pads the rest
    colors = {}
    sizes = []
    max_size = 0
    bbox_area = 0
    for o in objs:
        colors[o["color"]] = colors.get(o["color"], 0) + 1
        sizes.append(o["size"])
        max_size = max(max_size, o["size"])
        b = o.get("bbox") or (0, 0, 0, 0)
        bbox_area += max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
    # color entropy
    ent = 0.0
    for c in colors.values():
        p = c / n
        ent -= p * math.log(p + 1e-12)
    # vertical-mirror symmetry: fraction of objects with a same-color
    # counterpart mirrored across the horizontal center axis (within 4px)
    sym = 0
    for o in objs:
        mx = 63.0 - o["cx"]
        if any(p["color"] == o["color"]
               and abs(p["cy"] - o["cy"]) <= 4.0
               and abs(p["cx"] - mx) <= 4.0 for p in objs):
            sym += 1
    # overlap / touch pairs (normalized)
    ov = to = 0
    tot = 0
    for i in range(n):
        bi = objs[i].get("bbox") or (0, 0, 0, 0)
        for j in range(i + 1, n):
            tot += 1
            bj = objs[j].get("bbox") or (0, 0, 0, 0)
            # expand bi by 2px: intersect -> touch-or-overlap
            if not (bi[2] + 2 < bj[0] or bj[2] + 2 < bi[0]
                    or bi[3] + 2 < bj[1] or bj[3] + 2 < bi[1]):
                # true overlap (no margin)?
                if not (bi[2] < bj[0] or bj[2] < bi[0]
                        or bi[3] < bj[1] or bj[3] < bi[1]):
                    ov += 1
                else:
                    to += 1
    L = math.log1p
    return [
        L(n) / L(64.0),                       # 0 object count
        len(colors) / 16.0,                   # 1 color diversity
        ent / math.log(16.0),                # 2 color entropy
        L(max_size) / L(4096.0),             # 3 largest object
        (sum(L(s) for s in sizes) / n) / L(4096.0),  # 4 mean log size
        min(1.0, bbox_area / 4096.0),        # 5 bbox coverage
        sym / n,                              # 6 mirror symmetry
        ov / max(1, tot),                     # 7 overlap density
        to / max(1, tot),                     # 8 touch density
    ]


def featurize(objs, goals, hyps=None, inits=None, ctx=None):
    """Full 24-dim feature vector for a state.

    objs: object list (real or imagined).
    goals: the live GoalModule (for template progress).
    hyps/inits: precomputed hypothesis list + _init_counts per hyp id
        (planning path: bound once per call, reused for imagined states).
        None -> computed from objs here (logging path: real states).
    ctx: dict with static_anchors, regions, movers (from goals._analyze),
        actions_frac, arch_frac. None -> derived from goals (real states).
    """
    feats = [0.0] * N_FEATURES
    # -- template progress features (0-7): templates as *features* ------
    if hyps is None:
        hyps = goals.hypotheses(objs)
        inits = {id(h): goals._init_counts(h, objs) for h in hyps[:8]}
        hyps = hyps[:8]
    for h in hyps:
        t = TEMPLATE_INDEX.get(h["template"])
        if t is None:
            continue
        try:
            p = goals.progress(h, objs, inits.get(id(h)))
        except Exception:
            p = 0.0
        if p > feats[t]:
            feats[t] = min(1.0, max(0.0, p))
    # -- object features (8-16) ------------------------------------------
    of = _obj_features(objs)
    feats[8:17] = of[:9]
    # -- scene context features (17-21) ----------------------------------
    if ctx is None:
        an = goals._analyze()
        ctx = {"static_anchors": an["static_anchors"],
               "regions": an["regions"], "movers": an["movers"],
               "actions_frac": 0.0, "arch_frac": 0.0}
    n = len(objs)
    anchors = ctx.get("static_anchors") or []
    if n and anchors:
        ns = 0
        for o in objs:
            for (c, cy, cx, sz) in anchors:
                if (c == o["color"] and abs(cy - o["cy"]) <= 2.0
                        and abs(cx - o["cx"]) <= 2.0 and sz == o["size"]):
                    ns += 1
                    break
        feats[17] = ns / n  # static fraction
    regions = ctx.get("regions") or []
    feats[18] = 1.0 if regions else 0.0  # has visual target region
    movers = ctx.get("movers") or []
    if movers and regions and n:
        inner = [r["inner"] for r in regions]
        dmin = 90.0
        inside = 0
        nm = 0
        for o in objs:
            if o["color"] not in movers:
                continue
            nm += 1
            for r in inner:
                dy = 0.0
                if o["cy"] < r[0]:
                    dy = r[0] - o["cy"]
                elif o["cy"] > r[2]:
                    dy = o["cy"] - r[2]
                dx = 0.0
                if o["cx"] < r[1]:
                    dx = r[1] - o["cx"]
                elif o["cx"] > r[3]:
                    dx = o["cx"] - r[3]
                d = math.hypot(dy, dx)
                if d < dmin:
                    dmin = d
                if d == 0.0:
                    inside += 1
                    break
        feats[19] = min(1.0, dmin / 90.0)  # mover->region distance
        feats[20] = (inside / nm) if nm else 0.0  # movers inside regions
        feats[21] = min(1.0, len(movers) / 6.0)  # mover color count
    else:
        feats[19] = 1.0
    # -- trajectory context (22-23): constant during a planning call ------
    feats[22] = min(1.0, max(0.0, ctx.get("actions_frac", 0.0)))
    feats[23] = min(1.0, max(0.0, ctx.get("arch_frac", 0.0)))
    return feats


FEATURE_NAMES = (
    ["tmpl_%s" % t for t in
     ["inside_outline", "fill_outline", "overlap_static", "centroid_region",
      "touch_static", "cover_region", "eliminate", "clear_region"]]
    + ["n_objs", "n_colors", "color_entropy", "max_size", "mean_log_size",
       "bbox_coverage", "mirror_sym", "overlap_density", "touch_density",
       "static_frac", "has_region", "mover_region_dist",
       "movers_inside_frac", "n_movers", "actions_frac", "arch_frac"]
)


# ----------------------------------------------------------------------
# model: prior-centered ridge regression (numpy only)
# ----------------------------------------------------------------------

def _ridge_fit(X, y, w_prior, strength):
    """min ||Xw - y||^2 + strength * ||w - w_prior||^2 (closed form)."""
    d = X.shape[1]
    A = X.T @ X + strength * np.eye(d)
    b = X.T @ y + strength * w_prior
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


class SuccessModel:
    """V(s): learned progress-toward-completion.

    Cold start from the offline toy-trained prior; every level completion
    and game-over refits with prior-centered L2 (the prior counts as
    PRIOR_STRENGTH pseudo-samples). All numpy, deterministic, ms-scale.
    """

    def __init__(self, w_prior=None, strength=PRIOR_STRENGTH):
        self.w = np.zeros(N_FEATURES) if w_prior is None \
            else np.array(w_prior, dtype=float)
        if self.w.shape != (N_FEATURES,):
            self.w = np.zeros(N_FEATURES)
        self.w_prior = self.w.copy()
        self.strength = strength
        self.X = []  # list of (d,) online samples
        self.y = []
        self.refits = 0
        self.n_learned = 0  # samples since last refit

    @classmethod
    def from_prior(cls, path=PRIOR_PATH, strength=PRIOR_STRENGTH):
        try:
            with open(path) as f:
                p = json.load(f)
            w = p.get("weights")
            names = p.get("feature_names")
            if w and names == FEATURE_NAMES and len(w) == N_FEATURES:
                m = cls(w_prior=w, strength=strength)
                m.prior_info = p.get("trained_on", "?")
                return m
        except Exception:
            pass
        m = cls(w_prior=None, strength=strength)
        m.prior_info = "none (zero prior)"
        return m

    def add(self, feats_list, y_list):
        for x, y in zip(feats_list, y_list):
            self.X.append([float(v) for v in x])
            self.y.append(float(y))
        # cap memory: drop oldest
        if len(self.X) > MAX_DATA:
            self.X = self.X[-MAX_DATA:]
            self.y = self.y[-MAX_DATA:]
        self.n_learned += len(feats_list)

    def refit(self):
        if not self.X:
            return
        X = np.array(self.X)
        y = np.array(self.y)
        self.w = _ridge_fit(X, y, self.w_prior, self.strength)
        self.refits += 1
        self.n_learned = 0

    def value(self, feats):
        """V(s) in [0,1]. feats: raw 24-dim list."""
        v = float(np.dot(self.w, np.asarray(feats, dtype=float)))
        return min(1.0, max(0.0, v))

    def has_signal(self):
        """Has the model learned anything? Gates V usage in planning.

        With a zero prior and no online data, V(s) = 0 everywhere and the
        beam would be uninformed -- worse than template progress. Only
        climb V once there is something to climb.
        """
        return (getattr(self, "prior_info", "none") != "none (zero prior)"
                or self.refits > 0)

    def top_features(self, k=8):
        idx = np.argsort(-np.abs(self.w))[:k]
        return [(FEATURE_NAMES[i], float(self.w[i])) for i in idx]

    def summary(self):
        return {"prior": getattr(self, "prior_info", "?"),
                "samples": len(self.X), "refits": self.refits,
                "top_features": self.top_features(6)}


def save_prior(weights, trained_on, path=PRIOR_PATH):
    with open(path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES,
                   "weights": [float(w) for w in weights],
                   "trained_on": trained_on}, f, indent=1)
