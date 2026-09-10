"""Self-falsifying program world-models with MDL selection.

Hypothesis space H: per action-key, a small deterministic transition program:

    Rule = MODE  then  DELTA = {(x, y) -> color}

MODE is one of four tiny primitives (each a few integers of description):
  'none' : identity
  'all'  : SHIFT(dx, dy, bg)      -- translate the whole frame (camera motion)
  'fg'   : FGSHIFT(dx, dy, bg)    -- translate only non-background cells
                                     (sprite / cursor / avatar motion)
  'goto' : GOTO(click, bg)        -- translate non-bg cells so their centroid
                                     lands on the click position (click keys)

followed by a sparse absolute overwrite table DELTA ("these cells become
these colors"), which captures spawning, recoloring and redrawing effects.

Prediction:  pred(s) = apply_delta(apply_mode(s), delta)

Description length:  L(rule) = |delta| + 2*[shift used] + [mode != 'none']
  (each delta entry costs 1 token, a shift costs 2, choosing a non-trivial
  mode costs 1).

Data term: for a buffer B of observed (s, s') transitions under one action key,
  E(rule) = sum over pairs of |{cells: pred(s) != s'}|   (Hamming cell errors).

MDL score:  S(rule) = E(rule) + LAM * L(rule).

Learning = online falsification + evolution per action key:
  * after each real transition we test the current rule (prediction error);
  * if the error is surprisingly large (falsified) or enough new data arrived,
    we refit: fit a fresh rule from the buffer, generate mutants/crossovers of
    the current population, and keep the MDL-best (population size 3).

Only numpy is used. Everything runs on CPU in milliseconds.
"""

import numpy as np
from collections import deque

GRID = 32            # working resolution (64x64 frames coarsened 2x)
SHIFT_RANGE = 8      # candidate translations searched per refit
N_SHIFT_PAIRS = 4    # recent pairs used to score candidate shifts
POP_SIZE = 3         # programs kept per action key
N_MUTANTS = 6        # mutants generated per refit
BUF_CAP = 60         # transitions remembered per action key
LAM_DEFAULT = 5.0    # description-length weight (cell-errors per token)

MODES = ("none", "all", "c", "goto")


# ----------------------------------------------------------------------------
# grid helpers
# ----------------------------------------------------------------------------

def coarsen(frame64):
    """(64,64) int8 -> (32,32) int8 by 2x subsampling."""
    return np.ascontiguousarray(frame64[::2, ::2]).astype(np.int8)


def shift_grid(g, dx, dy, fill):
    """Translate whole grid: out[y+dy, x+dx] = g[y, x]; vacated cells = fill."""
    h, w = g.shape
    out = np.full((h, w), fill, dtype=np.int8)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    if xs1 > xs0 and ys1 > ys0:
        out[yd0:yd1, xd0:xd1] = g[ys0:ys1, xs0:xs1]
    return out


def shift_fg(g, dx, dy, bg):
    """Translate only non-bg cells by (dx, dy); bg cells stay put."""
    return shift_c(g, dx, dy, bg, None)


def shift_c(g, dx, dy, bg, colors, fill=None):
    """Translate only cells whose color is in `colors` (or all non-bg cells
    if colors is None); everything else stays put. Vacated cells become
    `fill` (default bg) -- the learned "what's behind the sprite"."""
    if fill is None:
        fill = bg
    h, w = g.shape
    if colors is None:
        fg = g != bg
    else:
        fg = np.isin(g, np.asarray(colors, dtype=np.int8))
    out = g.copy()
    out[fg] = fill
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        return out
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    out[ny[ok], nx[ok]] = g[ys[ok], xs[ok]]
    return out


def fg_centroid(g, bg):
    ys, xs = np.nonzero(g != bg)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def goto_click(g, qx, qy, bg):
    """Move non-bg cells so their centroid lands on (qx, qy)."""
    c = fg_centroid(g, bg)
    if c is None:
        return g.copy()
    return shift_fg(g, int(round(qx - c[0])), int(round(qy - c[1])), bg)


def mode_color(g):
    """Most common color in grid (background guess)."""
    bc = np.bincount(g.ravel(), minlength=16)
    return int(np.argmax(bc))


def frame_hash16(g32):
    """Coarse hash for novelty / win-memory: bytes of 16x16 subsample."""
    return g32[::2, ::2].tobytes()


def rhae(actions, baseline):
    """Relative action efficiency vs a baseline count, capped at 115."""
    a = max(1, actions)
    return min(((baseline / a) ** 2) * 100.0, 115.0)


# ----------------------------------------------------------------------------
# Rule: the "program"
# ----------------------------------------------------------------------------

class Rule:
    __slots__ = ("mode", "dx", "dy", "bg", "fill", "colors", "delta")

    def __init__(self, mode="none", dx=0, dy=0, bg=0, colors=(), delta=None,
                 fill=None):
        self.mode = mode
        self.dx = int(dx)
        self.dy = int(dy)
        self.bg = int(bg)
        self.colors = tuple(int(c) for c in colors)
        self.delta = dict(delta) if delta else {}  # (x, y) -> color
        self.fill = bg if fill is None else int(fill)

    def length(self):
        """Description length L(rule)."""
        L = len(self.delta)
        if self.mode in ("all", "c") and (self.dx or self.dy):
            L += 2
        if self.mode != "none":
            L += 1
        if self.mode == "c":
            L += len(self.colors)
            if self.fill != self.bg:
                L += 1
        return L

    def _apply_mode(self, g, click):
        if self.mode == "all":
            return shift_grid(g, self.dx, self.dy, self.bg)
        if self.mode == "c":
            return shift_c(g, self.dx, self.dy, self.bg, self.colors,
                           self.fill)
        if self.mode == "goto":
            if click is None:
                return g.copy()
            return goto_click(g, click[0], click[1], self.bg)
        return g.copy()

    def predict(self, g, click=None):
        out = self._apply_mode(g, click)
        for (x, y), c in self.delta.items():
            out[y, x] = c
        return out

    def errors(self, pairs):
        tot = 0
        for s, s2, click in pairs:
            tot += int(np.count_nonzero(self.predict(s, click) != s2))
        return tot

    def mdl(self, pairs, lam):
        return self.errors(pairs) + lam * self.length()

    def copy(self):
        return Rule(self.mode, self.dx, self.dy, self.bg, self.colors,
                    self.delta, self.fill)

    def __repr__(self):
        sh = f"({self.dx},{self.dy})" if self.mode in ("all", "c") else ""
        cc = f" cols={self.colors}" if self.mode == "c" else ""
        ff = f" fill={self.fill}" if self.mode == "c" and self.fill != self.bg else ""
        return (f"Rule({self.mode}{sh}{cc}{ff} bg={self.bg} "
                f"delta={len(self.delta)} L={self.length()})")


# ----------------------------------------------------------------------------
# fitting one rule from a transition buffer
# ----------------------------------------------------------------------------

def move_colors(pairs, bg):
    """Colors most involved in changes: candidates for the moving sprite."""
    ch = np.zeros(16, dtype=np.int64)
    for s, s2, _ in pairs:
        d = s != s2
        if d.any():
            ch += np.bincount(s[d].ravel(), minlength=16)
            ch += np.bincount(s2[d].ravel(), minlength=16)
    ch[bg] = 0
    total = ch.sum()
    if total == 0:
        return ()
    cols = [c for c in np.argsort(-ch) if ch[c] > 0.15 * total][:3]
    return tuple(int(c) for c in cols)


def _fit_delta(pairs, mode, dx, dy, bg, colors, fill, lam):
    """Best delta-template for a fixed (mode, shift, colors), computed exactly.

    Delta entries sit on distinct cells, so their error contributions are
    independent: entry (x,y,c) changes the total error by
        reduction = fix - hurt,
      fix  = #{pairs: base[y,x] != s2[y,x] = c}   (cell repaired)
      hurt = #{pairs: base[y,x] = s2[y,x] != c}    (cell broken)
    Keep the argmax color per cell iff reduction > lam (it pays for its
    description) and it fires consistently. Fully vectorized.
    """
    n = len(pairs)
    fix3 = np.zeros((GRID, GRID, 16), dtype=np.int32)
    right3 = np.zeros((GRID, GRID, 16), dtype=np.int32)
    proto = Rule(mode, dx, dy, bg, colors, fill=fill)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    cell = (yy * GRID + xx) * 16
    for s, s2, click in pairs:
        base = proto._apply_mode(s, click)
        code = (cell + s2).ravel()
        diff = (base != s2).ravel()
        np.add.at(fix3.ravel(), code[diff], 1)
        np.add.at(right3.ravel(), code[~diff], 1)
    total_right = right3.sum(axis=-1)                      # (32, 32)
    best_c = (fix3 + right3).argmax(axis=-1)                # (32, 32)
    fx = np.take_along_axis(fix3, best_c[..., None], -1)[..., 0]
    rc = np.take_along_axis(right3, best_c[..., None], -1)[..., 0]
    reduction = fx + rc - total_right
    thresh = max(2, (n + 1) // 2)
    keep = (reduction > lam) & (fx >= thresh)
    ys, xs = np.nonzero(keep)
    delta = {(int(x), int(y)): int(best_c[y, x])
             for y, x in zip(ys.tolist(), xs.tolist())}
    return Rule(mode, dx, dy, bg, colors, delta, fill)


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def est_shift(s, s2, colors):
    """Estimate translation of `colors` cells: centroid of vanished cells of
    those colors -> centroid of appeared cells of those colors."""
    cols = np.isin(s, np.asarray(colors, dtype=np.int8))
    van = cols & (s != s2)
    app = np.isin(s2, np.asarray(colors, dtype=np.int8)) & (s != s2)
    c1, c2 = _centroid(van), _centroid(app)
    if c1 is None or c2 is None:
        return None
    return int(round(c2[0] - c1[0])), int(round(c2[1] - c1[1]))


def est_fill(pairs, dx, dy, colors, bg):
    """What color is revealed behind the moving cells? Mode of s2 at the
    cells the moving colors vacated (excluding the moving colors)."""
    cols = []
    carr = np.asarray(colors, dtype=np.int8)
    for s, s2, _ in pairs[-8:]:
        m = np.isin(s, carr)
        if m.any():
            cols.extend(int(c) for c in s2[m].tolist() if c not in colors)
    if not cols:
        return bg
    return int(np.bincount(np.array(cols), minlength=16).argmax())


def _score_shift(pairs, mode, dx, dy, colors, bg, fill):
    err = 0
    for s, s2, ck in pairs:
        if mode == "all":
            base = shift_grid(s, dx, dy, bg)
        else:
            base = shift_c(s, dx, dy, bg, colors, fill)
        err += int(np.count_nonzero(base != s2))
        if err > 10 ** 9:
            break
    return err


def _shift_candidates(pairs, bg, click, movecolors):
    """Candidate (mode, dx, dy, colors, fill).

    'all': brute-force small search (camera motion). 'c': color sets from
    singletons + full set, shifts from centroid estimation +/- local search.
    """
    sub = pairs[-N_SHIFT_PAIRS:]
    out = []
    R = SHIFT_RANGE
    # -- whole-frame shifts (brute force) --
    scored = []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            scored.append((_score_shift(sub, "all", dx, dy, (), bg, bg),
                           dx, dy))
    scored.sort()
    for _, dx, dy in scored[:2]:
        out.append(("all", dx, dy, (), bg))
    # -- color shifts: singleton color sets + full set --
    color_sets = [(c,) for c in movecolors[:3]]
    if len(movecolors) > 1:
        color_sets.append(tuple(movecolors))
    s0, s2_0, _ = sub[-1]
    for colors in color_sets:
        est = est_shift(s0, s2_0, colors)
        tries = [(0, 0)]
        if est is not None:
            ex, ey = est
            if abs(ex) <= R and abs(ey) <= R:
                tries += [(ex + dx, ey + dy)
                          for dx in (-2, -1, 0, 1, 2)
                          for dy in (-2, -1, 0, 1, 2)]
        scored = sorted(((_score_shift(sub, "c", dx, dy, colors, bg, bg),
                           dx, dy) for dx, dy in tries))
        for _, dx, dy in scored[:2]:
            out.append(("c", dx, dy, colors, bg))
    out.append(("none", 0, 0, (), bg))
    if click is not None:
        out.append(("goto", 0, 0, (), bg))
    seen, uniq = set(), []
    for tup in out:
        if tup not in seen:
            seen.add(tup)
            uniq.append(tup)
    return uniq


def fit_rule(pairs, lam):
    """MDL-best Rule for a buffer: search mode/shift, fit delta, pick by S."""
    if not pairs:
        return Rule()
    bg = mode_color(pairs[-1][1])
    movecolors = move_colors(pairs, bg)
    click = pairs[-1][2]
    best, best_s = None, float("inf")
    for mode, dx, dy, colors, _fill in _shift_candidates(pairs, bg, click,
                                                        movecolors):
        fill = bg
        if mode == "c" and (dx or dy):
            fill = est_fill(pairs, dx, dy, colors, bg)
        r = _fit_delta(pairs, mode, dx, dy, bg, colors, fill, lam)
        sc = r.mdl(pairs, lam)
        if sc < best_s:
            best, best_s = r, sc
    return best


# ----------------------------------------------------------------------------
# mutation / crossover (evolutionary falsification loop)
# ----------------------------------------------------------------------------

def mutate(rule, rng, pairs):
    r = rule.copy()
    kind = int(rng.integers(N_MUTANTS + 2))
    if kind == 0:  # nudge shift
        r.dx = int(np.clip(r.dx + rng.integers(-1, 2), -SHIFT_RANGE, SHIFT_RANGE))
        r.dy = int(np.clip(r.dy + rng.integers(-1, 2), -SHIFT_RANGE, SHIFT_RANGE))
        if r.mode == "none":
            r.mode = "c"
            if pairs:
                r.colors = move_colors(pairs, r.bg)
    elif kind == 1:  # drop a random delta entry
        if r.delta:
            keys = list(r.delta.keys())
            del r.delta[keys[int(rng.integers(len(keys)))]]
    elif kind == 2:  # add an entry from the latest observed diff
        if pairs:
            s, s2, click = pairs[-1]
            base = r._apply_mode(s, click)
            ys, xs = np.nonzero(base != s2)
            if len(xs):
                i = int(rng.integers(len(xs)))
                r.delta[(int(xs[i]), int(ys[i]))] = int(s2[ys[i], xs[i]])
    elif kind == 3:  # re-estimate background
        if pairs:
            r.bg = mode_color(pairs[-1][1])
            if r.mode == "c":
                r.fill = est_fill(pairs, r.dx, r.dy, r.colors, r.bg)
    elif kind == 4:  # drop the delta table (pure motion)
        r.delta = {}
    elif kind == 5:  # switch motion mode
        r.mode = ("c", "all", "none")[int(rng.integers(3))]
        if r.mode == "c" and pairs:
            r.colors = move_colors(pairs, r.bg)
    elif kind == 6:  # perturb the moving-color set
        if pairs:
            mc = move_colors(pairs, r.bg)
            r.colors = mc
            if r.mode == "none" and mc:
                r.mode = "c"
    else:  # random small color-shift
        r.mode = "c"
        r.dx = int(rng.integers(-2, 3))
        r.dy = int(rng.integers(-2, 3))
        if pairs:
            r.colors = move_colors(pairs, r.bg)
    if r.mode == "c" and pairs and (r.dx or r.dy):
        r.fill = est_fill(pairs, r.dx, r.dy, r.colors, r.bg)
    return r


def crossover(r1, r2):
    inter = {k: v for k, v in r1.delta.items()
             if k in r2.delta and r2.delta[k] == v}
    union = dict(r1.delta)
    union.update(r2.delta)
    cols = r1.colors or r2.colors
    return [Rule(r1.mode, r1.dx, r1.dy, r1.bg, cols, inter, r1.fill),
            Rule(r1.mode, r1.dx, r1.dy, r2.bg, cols, union, r1.fill)]


# ----------------------------------------------------------------------------
# per-action-key model: buffer + population + falsification trigger
# ----------------------------------------------------------------------------

class KeyModel:
    """World-model for one action key: transition buffer + small population
    of candidate programs, evolved toward the MDL-best. `fallback` is an
    optional coarser model (used for click regions before they have data)."""

    def __init__(self, lam=LAM_DEFAULT, seed=0, fallback=None):
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.buf = deque(maxlen=BUF_CAP)  # (s, s2, click|None)
        self.pop = []
        self.rule = None
        self.fallback = fallback
        self.uses = 0          # informative (frame-changing) transitions
        self.tries = 0         # times this key was chosen (incl. no-ops)
        self.gameovers = 0
        self.ema_err = None
        self.refits = 0

    def note_gameover(self):
        self.gameovers += 1
        self.uses += 1

    @property
    def dangerous(self):
        if self.gameovers >= 2 and self.gameovers / max(1, self.uses) > 0.4:
            return True
        return self.fallback is not None and self.fallback.dangerous

    def note_prediction(self, s, s2, click=None):
        """Update the running prediction error without storing the pair
        (for transitions where nothing changed). Returns the error."""
        err = (0 if self.rule is None
               else int(np.count_nonzero(self.rule.predict(s, click) != s2)))
        self.ema_err = err if self.ema_err is None \
            else 0.9 * self.ema_err + 0.1 * err
        return err

    def observe(self, s, s2, click=None):
        err = self.note_prediction(s, s2, click)
        self.buf.append((s.copy(), s2.copy(), click))
        self.uses += 1

        falsified = (self.rule is not None and self.ema_err is not None
                     and err > max(12.0, 2.0 * self.ema_err))
        due = (self.rule is None and len(self.buf) >= 3) or (self.uses % 15 == 0)
        if falsified or due:
            self.refit()

    def refit(self):
        pairs = list(self.buf)
        if len(pairs) < 2:
            return
        cands = [fit_rule(pairs, self.lam)]
        for _ in range(N_MUTANTS):
            parent = self.pop[int(self.rng.integers(len(self.pop)))] \
                if self.pop else cands[0]
            cands.append(mutate(parent, self.rng, pairs))
        if len(self.pop) >= 2:
            cands += crossover(self.pop[0], self.pop[1])
        scored = sorted(((c.mdl(pairs, self.lam), c) for c in cands),
                        key=lambda t: t[0])
        seen, newpop = set(), []
        for _, c in scored:
            sig = (c.mode, c.dx, c.dy, c.bg, c.fill, c.colors,
                   tuple(sorted(c.delta.items())))
            if sig not in seen:
                seen.add(sig)
                newpop.append(c)
            if len(newpop) >= POP_SIZE:
                break
        self.pop = newpop
        self.rule = newpop[0]
        self.refits += 1

    def predict(self, g, click=None):
        if self.rule is not None:
            return self.rule.predict(g, click)
        if self.fallback is not None:
            return self.fallback.predict(g, click)
        return g
