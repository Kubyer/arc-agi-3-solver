"""Goal inference for the object-centric curiosity agent.

Curiosity learns *how* the world works but cannot infer *what the goal is*.
This module adds explicit goal hypotheses, tested against the only ground
truth available (levels_completed increments):

  1. Visual target detection: static elements are found from frames gathered
     during normal exploration (no extra probe actions -- the agent's
     curiosity already supplies a diverse action sequence). An *object* is
     static if an object of the same color stays at the same centroid with
     the same size in >=85% of observed frames (per-object, not per-color).
     Hollow rectangular outlines (a color forming a ring) are detected as
     target zones: their interiors are prime goal regions.
  2. Hypothesis generation: ~8 predicate templates over object relations,
     with parameters (colors, region rects) bound from the current scene and
     ranked by prior (static-region-involving first). Confirmed templates get
     a prior boost on later levels (transfer, parameterized by color/region
     role, never absolute positions).
  3. Hypothesis testing: the top untried hypothesis is turned into an action
     sequence by BFS over the *learned* transition model (model.predict_objects),
     executed, and scored: level completes -> confirmed; predicate achieved
     without completion -> refuted; plan fails to achieve the predicate ->
     inconclusive (stays untried); game over / surprise -> refuted.
     Refuted hypotheses are never retried on the same level.

Only numpy/scipy stdlib-level deps (numpy here). No per-game hardcoding.
"""

import math
import numpy as np
from collections import Counter, defaultdict, deque
from heapq import heappop, heappush

from .objects import signature

GOAL_K = 150        # stall threshold: no new archive cells for K steps
GOAL_M = 600        # periodic goal-mode check every M actions
GOAL_COOLDOWN = 120  # min actions between goal-mode evaluations
GOAL_EVERY = 12     # goal-directed interleave: 1 in 12 decisions pursues
                    # the top hypothesis best-effort (direction d)
N_HYPS_PER_EVAL = 3  # test up to this many hypotheses per evaluation
HINDSIGHT_WINDOW = 25  # confirm pursued hyps completed within this many
                       # actions before the level completed
PLAN_DEPTH = 5
PLAN_NODES = 600
PLAN_STOCH_DEPTH = 12   # stochastic beam search goes deeper (direction b)
PLAN_STOCH_BEAM = 24
PLAN_STOCH_SAMPLES = 3  # outcome samples per (state, key) expansion
PLAN_STOCH_NODES = 8000  # sampled-outcome budget per planning call
# Minimum predicted progress gain for launching a *partial*-plan pursuit
# (a full plan that achieves the predicate always launches as a test).
PURSUIT_GAIN_MIN = 0.15
# Cap on sustained pursuit plan length (interleave / evaluation).
PURSUIT_PLAN_CAP = 6
MIN_MODEL_KEYS = 2  # need this many well-observed keys before testing
MIN_MODEL_N = 3     # ...with at least this many observations each

# Template -> feature index in success.py's learned-signal vector.
TEMPLATE_INDEX = {
    "inside_outline": 0, "fill_outline": 1, "overlap_static": 2,
    "centroid_region": 3, "touch_static": 4, "cover_region": 5,
    "eliminate": 6, "clear_region": 7,
}


# ----------------------------------------------------------------------
# geometry helpers (rects are (y0, x0, y1, x1) inclusive, like objects.py)
# ----------------------------------------------------------------------

def _in_rect(o, r, margin=0):
    return (r[0] - margin <= o["cy"] <= r[2] + margin and
            r[1] - margin <= o["cx"] <= r[3] + margin)


def _rect_center(r):
    return ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)


def _bbox(o):
    b = o.get("bbox")
    if b and not (b[0] == 0 and b[1] == 0 and b[2] == 0 and b[3] == 0):
        return b
    cy, cx = int(round(o["cy"])), int(round(o["cx"]))
    return (cy - 1, cx - 1, cy + 1, cx + 1)


def _area(r):
    return max(0, r[2] - r[0] + 1) * max(0, r[3] - r[1] + 1)


def _intersect(a, b):
    y0 = max(a[0], b[0])
    x0 = max(a[1], b[1])
    y1 = min(a[2], b[2])
    x1 = min(a[3], b[3])
    if y1 < y0 or x1 < x0:
        return None
    return (y0, x0, y1, x1)


def _overlap_frac(a, b):
    inter = _intersect(_bbox(a), _bbox(b))
    if inter is None:
        return 0.0
    denom = min(_area(_bbox(a)), _area(_bbox(b)))
    return _area(inter) / denom if denom else 0.0


def _touch(a, b):
    ba, bb = _bbox(a), _bbox(b)
    ea = (ba[0] - 2, ba[1] - 2, ba[2] + 2, ba[3] + 2)
    if _intersect(ea, bb) is None:
        return False
    # adjacent, not deeply overlapping
    return _overlap_frac(a, b) < 0.5


def _contains(outer, inner, tol=0):
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3])


def _shrink(r, m):
    return (r[0] + m, r[1] + m, r[2] - m, r[3] - m)


def _dist_to_rect(cy, cx, r):
    """Euclidean distance from a point to a rect (0 if inside)."""
    dy = 0.0
    if cy < r[0]:
        dy = r[0] - cy
    elif cy > r[2]:
        dy = cy - r[2]
    dx = 0.0
    if cx < r[1]:
        dx = r[1] - cx
    elif cx > r[3]:
        dx = cx - r[3]
    return math.hypot(dy, dx)


def fine_hash(objs):
    """Fine-grained state hash for the planner's visited set.

    Unlike objects.signature (8px bins, built for a compact Go-Explore
    archive), this uses 2px bins so small precise moves produce distinct
    imagined states -- the planner can then represent and aim for fine
    progress that the coarse signature would collapse (direction c's
    precision concern, without exploding the archive).
    """
    return tuple(sorted(
        (o["color"], round(o["cy"] / 2.0), round(o["cx"] / 2.0),
         o["size_bin"])
        for o in objs
    ))


# ----------------------------------------------------------------------
# visual target detection
# ----------------------------------------------------------------------

def detect_outlines(frame, objs):
    """Find hollow rectangular outlines: a color forming a ring.

    An object qualifies if it is large, its bounding-box fill ratio is low
    (ring, not solid), and the bbox interior (shrunk by 2px) contains no
    pixels of its color. Returns [{color, bbox, inner}]; inner is the target
    region rect.
    """
    frame = np.asarray(frame)
    out = []
    for o in objs:
        y0, x0, y1, x1 = o["bbox"]
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if h < 8 or w < 8 or o["size"] < 40:
            continue
        fill = o["size"] / (h * w)
        if not (0.08 <= fill <= 0.85):
            continue
        inner = _shrink(o["bbox"], 2)
        if inner[2] <= inner[0] or inner[3] <= inner[1]:
            continue
        sub = frame[inner[0]:inner[2] + 1, inner[1]:inner[3] + 1]
        if np.any(sub == o["color"]):
            continue
        out.append({"color": o["color"], "bbox": o["bbox"], "inner": inner})
    return out


# ----------------------------------------------------------------------
# goal module
# ----------------------------------------------------------------------

class GoalModule:
    def __init__(self):
        # transfer state: persists across levels of a game
        self.template_wins = Counter()
        self.confirmed = []   # [(level_idx, hyp_id)]
        self.tests = 0
        self._analyze_cache = {}  # len(frames) -> analysis (frames append-only)
        self.new_level()

    # -- per-level state ------------------------------------------------

    def new_level(self):
        self.frames = []        # parsed object lists observed this level
        self.outlines = []      # hollow-outline target zones
        self._outline_tried = 0
        self.scoreboard = {}    # hyp_id -> {"status": ...}
        self.testing = None     # hyp_id currently under test
        self.testing_hyp = None  # full hypothesis dict under test
        self.pursuit_hyp = None  # hyp pursued best-effort (no full plan)
        self.pursued = []       # [(action_idx, hyp)] pursued this level,
                                # for hindsight credit on completion
        self._plan_fails = Counter()
        self._analyze_cache = {}

    def observe(self, objs, frame):
        if len(self.frames) < 100:
            self.frames.append(list(objs))
        # outline detection needs a trustworthy frame: skip the first two
        # (a mid-game level change can leave one stale frame in the pipe).
        if self._outline_tried < 2 and len(self.frames) in (3, 6):
            self._outline_tried += 1
            det = detect_outlines(frame, objs)
            if det:
                self.outlines = det
                self._outline_tried = 2

    # -- static / movable analysis --------------------------------------

    def _analyze(self):
        """Cached static/movable decomposition.

        self.frames is append-only within a level (capped at 100), so the
        analysis is a pure function of len(frames) -- and so is
        self.outlines, which is set from observe() at fixed frame counts.
        Cache on len(frames); hypotheses() calls this repeatedly per
        decision, and success.py's feature logging calls it every few
        steps, so the cache matters.
        """
        key = len(self.frames)
        an = self._analyze_cache.get(key)
        if an is None:
            an = self._analyze_uncached()
            # keep the cache small; only recent lengths matter
            if len(self._analyze_cache) > 8:
                self._analyze_cache.pop(
                    next(iter(self._analyze_cache)))
            self._analyze_cache[key] = an
        return an

    def _analyze_uncached(self):
        """Static vs movable decomposition of the scene.

        Per-OBJECT (not per-color): an object is static if an object of the
        same color stays at the same centroid with the same size in nearly
        every observed frame. Frames 0-1 are excluded -- a mid-game level
        change can leave one stale frame in the pipe. Static objects with
        size >= 40 become candidate target regions (in addition to hollow
        outlines, which are detected separately).

        Recomputed from scratch on every call (frames <= 100, so this is
        cheap); hypotheses always bind to the latest scene.
        """
        frames = self.frames
        n = len(frames)
        # anchor on frame 2 (first guaranteed trustworthy frame); match
        # against frames 2..n-1 excluding the anchor itself.
        anchor_i = 2 if n > 2 else 0
        anchor = frames[anchor_i] if n else []
        static_anchors = []  # (color, cy, cx, size)
        if n >= 6:
            others = [fr for i, fr in enumerate(frames)
                      if i not in (0, 1, anchor_i)]
            for o in anchor:
                if not others:
                    break
                match = sum(
                    1 for fr in others
                    if any(p["color"] == o["color"]
                           and abs(p["cy"] - o["cy"]) <= 2.0
                           and abs(p["cx"] - o["cx"]) <= 2.0
                           and p["size"] == o["size"] for p in fr))
                if match / len(others) >= 0.85:
                    static_anchors.append((o["color"], o["cy"], o["cx"],
                                           o["size"]))

        def is_static(o):
            return any(c == o["color"] and abs(cy - o["cy"]) <= 2.0
                       and abs(cx - o["cx"]) <= 2.0 and sz == o["size"]
                       for c, cy, cx, sz in static_anchors)

        latest = frames[-1] if frames else []
        regions = []
        for ol in self.outlines:
            regions.append({"kind": "outline", "color": ol["color"],
                            "rect": ol["bbox"], "inner": ol["inner"]})
        for o in latest:
            if o["size"] >= 40 and is_static(o):
                if any(r["color"] == o["color"] for r in regions):
                    continue  # outline (or other) region already covers it
                inner = _shrink(o["bbox"], 2)
                if inner[2] <= inner[0] or inner[3] <= inner[1]:
                    inner = o["bbox"]  # thin object: region is the bbox
                regions.append({"kind": "static", "color": o["color"],
                                "rect": o["bbox"], "inner": inner})
        regions = regions[:4]
        region_colors = {r["color"] for r in regions}
        movers = []
        for o in latest:
            # A region color (large static structure / outline) is never a
            # mover, even if a small fragment of that color shifts: binding
            # a "move X" hypothesis to a mostly-static color produces
            # predicates that can never be satisfied (e.g. overlapping two
            # walls) and wastes pursuits on them.
            if (not is_static(o) and o["color"] not in movers
                    and o["color"] not in region_colors):
                movers.append(o["color"])
        movers.sort(key=lambda c: -sum(p["size"] for p in latest
                                       if p["color"] == c))
        movers = movers[:6]
        return {"static_anchors": static_anchors,
                "regions": regions, "movers": movers}

    # -- hypothesis generation ------------------------------------------

    def hypotheses(self, objs):
        """All candidate hypotheses bound to the current scene."""
        an = self._analyze()
        hyps = []

        def add(template, params, prior):
            hid = (template, tuple(sorted(params.items())))
            st = self.scoreboard.get(hid, {}).get("status", "untried")
            hyps.append({"id": hid, "template": template,
                         "params": dict(params), "prior": prior,
                         "status": st})

        for mc in an["movers"]:
            for rg in an["regions"]:
                inner = rg["inner"]
                if rg["kind"] == "outline":
                    add("inside_outline",
                        {"mover": mc, "region": rg["color"], "rect": inner},
                        1.00)
                    add("fill_outline",
                        {"mover": mc, "region": rg["color"], "rect": inner},
                        0.90)
                add("overlap_static",
                    {"mover": mc, "target": rg["color"]}, 0.80)
                add("centroid_region",
                    {"mover": mc, "region": rg["color"], "rect": inner}, 0.75)
                add("touch_static",
                    {"mover": mc, "target": rg["color"]}, 0.60)
                add("cover_region",
                    {"mover": mc, "region": rg["color"], "rect": inner}, 0.55)
            add("eliminate", {"color": mc}, 0.50)
        for rg in an["regions"]:
            add("clear_region",
                {"region": rg["color"], "rect": rg["inner"],
                 "movers": tuple(sorted(an["movers"]))}, 0.50)
        return hyps

    # -- predicate checking ----------------------------------------------

    def satisfied(self, hyp, objs):
        t = hyp["template"]
        p = hyp["params"]
        by_color = defaultdict(list)
        for o in objs:
            by_color[o["color"]].append(o)
        if t == "inside_outline":
            r = p["rect"]
            return any(_in_rect(o, r) for o in by_color.get(p["mover"], []))
        if t == "fill_outline":
            r = p["rect"]
            area = max(1, _area(r))
            cov = (sum(o["size"] for o in by_color.get(p["mover"], [])
                       if _in_rect(o, r)) / area)
            return cov >= 0.6
        if t == "overlap_static":
            return any(_overlap_frac(a, b) > 0.15
                       for a in by_color.get(p["mover"], [])
                       for b in by_color.get(p["target"], []))
        if t == "centroid_region":
            ms = by_color.get(p["mover"], [])
            if not ms:
                return False
            o = max(ms, key=lambda o: o["size"])
            cy, cx = _rect_center(p["rect"])
            return abs(o["cy"] - cy) <= 4 and abs(o["cx"] - cx) <= 4
        if t == "touch_static":
            return any(_touch(a, b)
                       for a in by_color.get(p["mover"], [])
                       for b in by_color.get(p["target"], []))
        if t == "cover_region":
            return any(_contains(_bbox(o), p["rect"], tol=2)
                       for o in by_color.get(p["mover"], []))
        if t == "eliminate":
            return not by_color.get(p["color"])
        if t == "clear_region":
            r = p["rect"]
            return not any(_in_rect(o, r)
                           for c in p["movers"]
                           for o in by_color.get(c, []))
        return False

    # -- ranking / scoreboard ---------------------------------------------

    def _rank(self, h):
        return (h["prior"] + 0.5 * self.template_wins[h["template"]],
                h["prior"])

    def _manipulable(self, color, model, for_removal=False):
        """Does a well-observed key plausibly manipulate this color?

        Sharper hypothesis generation: don't waste tests on colors no key
        is known to affect. A direct click key on the color counts (clicks
        are handles by construction); otherwise require observed mean
        movement (placement) or a high disappearance rate (removal).
        """
        for k, st in model.stats.items():
            if st.get("n", 0) < MIN_MODEL_N or st.get("fatal", 0):
                continue
            if k == "o%d" % color:
                return True
            if for_removal:
                gone = sum(c for (c2, _sb), c in st["disappear"].items()
                           if c2 == color)
                if gone / st["n"] > 0.3:
                    return True
            else:
                ds = st["move"].get(color)
                if ds:
                    my = sum(d[0] for d in ds) / len(ds)
                    mx = sum(d[1] for d in ds) / len(ds)
                    if abs(my) > 0.5 or abs(mx) > 0.5:
                        return True
        return False

    def _grounded(self, hyp, model):
        """Hypothesis's actor color is manipulable by a known key."""
        t = hyp["template"]
        p = hyp["params"]
        if t == "eliminate":
            return self._manipulable(p["color"], model, for_removal=True)
        if t == "clear_region":
            return any(self._manipulable(c, model, for_removal=True)
                       for c in p["movers"])
        return self._manipulable(p["mover"], model)

    def top_hypothesis(self, objs, model=None, region_only=False,
                       require_manipulable=False):
        """Best untried hypothesis, or None.

        Hypotheses already true in the current frame are refuted for free:
        if the predicate held and the level is not complete, it cannot be
        the goal.

        region_only: only hypotheses bound to a detected visual target
        (outline/static region). Used for goal-directed interleaving: with
        no visual target there is nothing concrete to aim best-effort
        actions at, and this keeps interleaving a no-op on games where no
        target was ever detected (e.g. the toys -> bit-identical).
        require_manipulable: skip hypotheses whose actor color no known key
        affects (needs model).
        """
        cands = []
        for h in self.hypotheses(objs):
            if h["status"] != "untried":
                continue
            if self._plan_fails[h["id"]] >= 3:
                continue
            if region_only and h["template"] == "eliminate":
                continue
            if (require_manipulable and model is not None
                    and not self._grounded(h, model)):
                continue
            if self.satisfied(h, objs):
                self.scoreboard[h["id"]] = {"status": "refuted",
                                            "reason": "already-true"}
                continue
            cands.append(h)
        if not cands:
            return None
        cands.sort(key=self._rank, reverse=True)
        return cands[0]

    # -- planning over the learned transition model ----------------------

    def plan_for(self, hyp, objs, model, legal):
        """BFS over model.predict_objects for a key sequence predicted to
        satisfy hyp. Returns [keys] or None."""
        if self.satisfied(hyp, objs):
            return None
        keys = [k for k in legal
                if model.stats.get(k, {}).get("n", 0) >= 1
                and not model.stats[k].get("fatal", 0)]
        keys = sorted(keys, key=lambda k: -model.stats[k]["n"])[:12]
        if not keys:
            return None
        seen = {signature(objs)}
        dq = deque([(list(objs), [])])
        nodes = 0
        while dq and nodes < PLAN_NODES:
            cur, plan = dq.popleft()
            if len(plan) >= PLAN_DEPTH:
                continue
            for k in keys:
                po, _added = model.predict_objects(k, cur)
                nodes += 1
                if self.satisfied(hyp, po):
                    return plan + [k]
                ps = signature(po)
                if ps in seen:
                    continue
                seen.add(ps)
                dq.append((po, plan + [k]))
        return None

    # -- predicate progress (heuristic for deeper search) -----------------

    def _init_counts(self, hyp, objs):
        """Reference counts in the planning-root state for progress()."""
        t = hyp["template"]
        p = hyp["params"]
        init = {}
        if t == "eliminate":
            init[("elim", p["color"])] = sum(1 for o in objs
                                            if o["color"] == p["color"])
        elif t == "clear_region":
            init[("clear", p["region"])] = sum(
                1 for c in p["movers"] for o in objs
                if o["color"] == c and _in_rect(o, p["rect"]))
        return init

    @staticmethod
    def _mover(hyp, objs):
        ms = [o for o in objs if o["color"] == hyp["params"]["mover"]]
        return max(ms, key=lambda o: o["size"]) if ms else None

    def progress(self, hyp, objs, init=None):
        """Smooth [0,1] progress toward hyp's predicate (1 = satisfied).

        Gives the stochastic planner a gradient to climb so best-first
        search can go deeper than the deterministic BFS: partial credit for
        moving the actor toward its target region, growing coverage, or
        removing target objects.
        """
        if self.satisfied(hyp, objs):
            return 1.0
        t = hyp["template"]
        p = hyp["params"]
        by_color = defaultdict(list)
        for o in objs:
            by_color[o["color"]].append(o)
        if t in ("inside_outline", "centroid_region", "cover_region"):
            m = self._mover(hyp, objs)
            if m is None:
                return 0.0
            return max(0.0, 1.0 - _dist_to_rect(m["cy"], m["cx"],
                                               p["rect"]) / 64.0)
        if t == "fill_outline":
            area = max(1, _area(p["rect"]))
            cov = (sum(o["size"] for o in by_color.get(p["mover"], [])
                       if _in_rect(o, p["rect"])) / area)
            return min(1.0, cov / 0.6)
        if t in ("overlap_static", "touch_static"):
            ms = by_color.get(p["mover"], [])
            ts = by_color.get(p["target"], [])
            if not ms or not ts:
                return 0.0
            d = min(math.hypot(a["cy"] - b["cy"], a["cx"] - b["cx"])
                    for a in ms for b in ts)
            return max(0.0, 1.0 - d / 64.0)
        if t == "eliminate":
            c = p["color"]
            rem = sum(1 for o in objs if o["color"] == c)
            tot = (init or {}).get(("elim", c), max(rem, 1))
            return 1.0 - rem / max(1, tot)
        if t == "clear_region":
            r = p["rect"]
            rem = sum(1 for c in p["movers"] for o in by_color.get(c, [])
                      if _in_rect(o, r))
            tot = (init or {}).get(("clear", p["region"]), max(rem, 1))
            return 1.0 - rem / max(1, tot)
        return 0.0

    # -- stochastic planning over outcome distributions -------------------

    def plan_beam(self, hyp, objs, model, legal, rng,
                  beam=PLAN_STOCH_BEAM, depth=PLAN_STOCH_DEPTH,
                  samples=PLAN_STOCH_SAMPLES, node_budget=PLAN_STOCH_NODES,
                  score_fn=None):
        """Beam search over *sampled* model outcomes toward hyp.

        Direction (a)+(b): instead of BFS over per-key *mean* displacements
        (depth<=5, which smooths away the distinct outcomes a key actually
        produces), expand a best-first beam over draws from the empirical
        per-key outcome distribution (model.sample_objects), scored by
        predicate progress + novelty, with a fine-grained visited set.

        score_fn (round 3): when given, imagined states are scored by
        score_fn(state) -- the learned success signal V(s) -- instead of
        template progress. The *achieved* verdict stays predicate-based
        (formal tests need falsifiable structure), but the beam climbs
        the learned target. gain is then measured in score_fn units.

        Returns (plan, achieved, gain): plan is the key sequence, achieved
        is whether its end state satisfies the predicate (validated by
        Monte-Carlo rollouts), gain is progress(end) - progress(root).
        When no validated full plan exists, plan is the best *partial*
        plan found (highest progress) -- used for sustained goal-directed
        pursuit, which can compose the multi-step sequences that single
        best-effort steps cannot.
        """
        if self.satisfied(hyp, objs):
            return ([], False, 0.0)
        keys = [k for k in legal
                if model.stats.get(k, {}).get("n", 0) >= 1
                and not model.stats[k].get("fatal", 0)]
        keys = sorted(keys, key=lambda k: -model.stats[k]["n"])[:12]
        if not keys:
            return ([], False, 0.0)
        init = self._init_counts(hyp, objs)
        root = [dict(o) for o in objs]
        _sc = score_fn if score_fn is not None else \
            (lambda s: self.progress(hyp, s, init))
        root_prog = _sc(root)
        # heap of (-score, tiebreak, plan, state); best-first by progress
        # plus a novelty bonus for imagined states never seen before.
        heap = [(-(root_prog + 0.3), 0, [], root)]
        seen = {fine_hash(root)}
        nodes, tie = 0, 0
        best_plan, best_prog = [], root_prog
        while heap and nodes < node_budget:
            _, _, plan, cur = heappop(heap)
            if len(plan) >= depth:
                continue
            for k in keys:
                for _ in range(samples):
                    po, _added = model.sample_objects(k, cur, rng)
                    nodes += 1
                    if self.satisfied(hyp, po):
                        cand = plan + [k]
                        if self._validate_plan(hyp, root, cand, model,
                                               init, rng):
                            return (cand, True, 1.0 - root_prog)
                        # lucky branch that doesn't reproduce: don't
                        # return it, and don't re-expand this state.
                        seen.add(fine_hash(po))
                        continue
                    fh = fine_hash(po)
                    if fh in seen:
                        continue
                    seen.add(fh)
                    prog = _sc(po)
                    if prog > best_prog + 1e-9:
                        best_prog, best_plan = prog, plan + [k]
                    tie += 1
                    heappush(heap, (-(prog + 0.3), tie, plan + [k], po))
            if len(heap) > beam * 4:
                # keep the beam bounded: drop the worst imagined states
                heap = sorted(heap, key=lambda e: e[0])[:beam]
        return (best_plan, False, best_prog - root_prog)

    def plan_value(self, objs, model, legal, rng, value_fn,
                   beam=PLAN_STOCH_BEAM, depth=PLAN_STOCH_DEPTH,
                   samples=PLAN_STOCH_SAMPLES,
                   node_budget=PLAN_STOCH_NODES):
        """Template-free beam search maximizing the learned value.

        Round 3: for games where no hypothesis is plannable (no detected
        visual target, templates exhausted), the planner still needs
        something worth pursuing. This searches purely over value_fn --
        the learned success signal V(s) -- with no predicate and no test
        semantics: the returned plan is always a *pursuit* (never
        refuted), composing multi-step sequences toward success-like
        states. Returns (plan, gain); gain <= 0 means "don't bother".
        """
        keys = [k for k in legal
                if model.stats.get(k, {}).get("n", 0) >= 1
                and not model.stats[k].get("fatal", 0)]
        keys = sorted(keys, key=lambda k: -model.stats[k]["n"])[:12]
        if not keys:
            return ([], 0.0)
        root = [dict(o) for o in objs]
        root_v = value_fn(root)
        heap = [(-root_v, 0, [], root)]
        seen = {fine_hash(root)}
        nodes, tie = 0, 0
        best_plan, best_v = [], root_v
        while heap and nodes < node_budget:
            _, _, plan, cur = heappop(heap)
            if len(plan) >= depth:
                continue
            for k in keys:
                for _ in range(samples):
                    po, _added = model.sample_objects(k, cur, rng)
                    nodes += 1
                    fh = fine_hash(po)
                    if fh in seen:
                        continue
                    seen.add(fh)
                    v = value_fn(po)
                    if v > best_v + 1e-9:
                        best_v, best_plan = v, plan + [k]
                    tie += 1
                    heappush(heap, (-v, tie, plan + [k], po))
            if len(heap) > beam * 4:
                heap = sorted(heap, key=lambda e: e[0])[:beam]
        return (best_plan, best_v - root_v)

    def plan_stochastic(self, hyp, objs, model, legal, rng,
                        beam=PLAN_STOCH_BEAM, depth=PLAN_STOCH_DEPTH,
                        samples=PLAN_STOCH_SAMPLES,
                        node_budget=PLAN_STOCH_NODES):
        """Validated full plan toward hyp, or None (see plan_beam)."""
        plan, achieved, _gain = self.plan_beam(
            hyp, objs, model, legal, rng, beam, depth, samples, node_budget)
        return plan if achieved else None

    def _validate_plan(self, hyp, root, plan, model, init, rng, trials=8,
                       need=3):
        """Monte-Carlo check: does this plan achieve the predicate
        robustly, or was it a lucky sample branch? Execute the key
        sequence `trials` times under fresh draws; require `need`
        successes. Filters plans that would fail open-loop execution
        (which wastes the test and yields only 'inconclusive').
        """
        ok = 0
        for _ in range(trials):
            cur = [dict(o) for o in root]
            for k in plan:
                cur, _a = model.sample_objects(k, cur, rng)
            if self.satisfied(hyp, cur):
                ok += 1
                if ok >= need:
                    return True
        return False

    def best_effort_key(self, hyp, objs, model, legal, rng, samples=4):
        """Single key maximizing expected one-step predicate progress.

        For goal-directed interleaving when no full plan exists: a partial
        step toward the top hypothesis beats another curiosity probe when
        the hypothesis is right. Returns None when no key improves on the
        current progress (don't waste the action).
        """
        keys = [k for k in legal
                if model.stats.get(k, {}).get("n", 0) >= 1
                and not model.stats[k].get("fatal", 0)]
        keys = sorted(keys, key=lambda k: -model.stats[k]["n"])[:12]
        if not keys:
            return None
        init = self._init_counts(hyp, objs)
        base = self.progress(hyp, objs, init)
        best, bestv = None, base + 0.02
        order = list(keys)
        rng.shuffle(order)  # random tie-break to avoid lock-in
        for k in order:
            v = sum(self.progress(
                hyp, model.sample_objects(k, objs, rng)[0], init)
                for _ in range(samples)) / samples
            if v > bestv:
                best, bestv = k, v
        return best

    # -- test lifecycle ---------------------------------------------------

    def begin_test(self, hyp):
        self.testing = hyp["id"]
        self.testing_hyp = hyp
        self.scoreboard[hyp["id"]] = {"status": "testing"}
        self.tests += 1

    def begin_pursuit(self, hyp, action_idx):
        """Best-effort goal-directed action (no full plan). Lighter than a
        test: recorded for hindsight credit, but a non-achieving step is
        inconclusive and the hypothesis stays untried."""
        self.pursuit_hyp = hyp
        self.pursued.append((action_idx, hyp))

    def abort_pursuit(self):
        """End pursuit without verdict (inconclusive)."""
        self.pursuit_hyp = None

    def hindsight(self, objs, level_idx, action_idx,
                  window=HINDSIGHT_WINDOW):
        """Wider confirmation attribution (direction: credit near-misses).

        On level completion with no in-flight plan test: confirm the
        top-ranked *pursued* hypothesis whose predicate holds in the
        winning frame, provided it was pursued within `window` actions of
        the completion. The completion is only detected at the top of the
        next loop iteration, so without this, goal-directed work that paid
        off is never credited.
        """
        if self.testing is not None:
            return None
        cands = [h for ai, h in self.pursued
                 if action_idx - ai <= window
                 and self.scoreboard.get(h["id"], {}).get(
                     "status", "untried") == "untried"
                 and self.satisfied(h, objs)]
        if not cands:
            return None
        cands.sort(key=self._rank, reverse=True)
        h = cands[0]
        self.scoreboard[h["id"]] = {"status": "confirmed",
                                    "reason": "hindsight"}
        self.template_wins[h["template"]] += 1
        self.confirmed.append((level_idx, h["id"]))
        return h

    def note_plan_fail(self, hid):
        self._plan_fails[hid] += 1

    def _clear_test(self):
        self.testing = None
        self.testing_hyp = None

    def confirm(self, level_idx):
        hid = self.testing
        if hid is None:
            return
        self.scoreboard[hid] = {"status": "confirmed"}
        self.template_wins[hid[0]] += 1
        self.confirmed.append((level_idx, hid))
        self._clear_test()

    def refute(self, reason):
        hid = self.testing
        if hid is None:
            return
        self.scoreboard[hid] = {"status": "refuted", "reason": reason}
        self._clear_test()

    def abort(self):
        """Cut a test short without verdict (e.g. level-cap reset)."""
        hid = self.testing
        if hid is None:
            self.pursuit_hyp = None
            return
        self.scoreboard[hid] = {"status": "untried"}
        self._clear_test()
        self.pursuit_hyp = None

    def on_game_over(self):
        self.refute("game-over")

    def summary(self):
        counts = Counter(v["status"] for v in self.scoreboard.values())
        return {"tests": self.tests, "confirmed": len(self.confirmed),
                "pursuits": len(self.pursued),
                "scoreboard": dict(counts),
                "confirmed_detail": [(lvl, hid[0],
                                      dict(hid[1])) for lvl, hid in
                                     self.confirmed]}
