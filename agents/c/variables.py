"""Causal variables: parse a (64,64) int8 frame into a small set of binned
object-level variables for the structural causal model.

Variable set (17, all binned to small integer ranges):
  ncol  - number of distinct non-bg colors, 0..6
  fg    - total non-bg pixels, log2-ish bins 0..9
  d12   - min Manhattan distance between slot-1 and slot-2 pixels, 0..3
          (0=touching, 1=adjacent, 2=near, 3=far/absent)
  gcx   - centroid x of the largest foreground object, 16 bins
  gcy   - centroid y of the largest foreground object, 16 bins
  per color slot k=1..6 (slots = first-seen non-bg colors, stable identity):
    cnt_k - pixel count of slot color, fine log bins 0..9
    obj_k - connected-component count (4-conn), 0..15 (fine: object count is
            the key causal quantity -- one action often adds exactly one)

Slot identity is first-seen order (stable across frames), NOT per-frame rank:
this keeps "cnt_2" referring to the same color over time, which is what the
causal discovery needs for cross-context invariance. Six slots are kept so
that click interventions cover games with 5+ colors; per-slot position was
dropped (replaced by the global largest-object centroid) to stay under the
~20-variable budget.

Binning lesson: coarse bins destroy the intervention signal (a 270-pixel
change left every variable unmoved). Bins are fine enough that typical
single-action effects register, at the cost of a larger state space --
sparsity is handled by pooling (most-likely-delta) in the causal model.
"""

import numpy as np
from collections import deque

N_SLOTS = 6
CX_BINS = 16  # centroid bins per axis


def _var_names():
    # 17 causal variables (task budget: under ~20):
    #   globals: ncol, fg, d12, gcx, gcy
    #   per color slot (6 slots x 2 attrs): cnt, obj
    # Six slots are kept (not four) because click interventions are
    # enumerated from slots -- a game with 5+ colors (e.g. tn36) needs all
    # six to be clickable. Position is tracked globally (largest foreground
    # object's centroid) instead of per-slot to stay within budget.
    names = ["ncol", "fg", "d12", "gcx", "gcy"]
    for k in range(1, N_SLOTS + 1):
        names += [f"cnt{k}", f"obj{k}"]
    return names


VAR_NAMES = _var_names()

# max bin value per variable (for clipping in forward simulation)
BIN_MAX = {"ncol": 6, "fg": 9, "d12": 3, "gcx": CX_BINS - 1,
           "gcy": CX_BINS - 1}
for k in range(1, N_SLOTS + 1):
    BIN_MAX[f"cnt{k}"] = 9
    BIN_MAX[f"obj{k}"] = 15


def _bin_count(n: int) -> int:
    """Fine log-ish bins: 0,1,2-3,4-7,8-15,16-31,32-63,64-127,128-255,256+."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    if n <= 7:
        return 3
    if n <= 15:
        return 4
    if n <= 31:
        return 5
    if n <= 63:
        return 6
    if n <= 127:
        return 7
    if n <= 255:
        return 8
    return 9


def _bin_obj(n: int) -> int:
    # fine bins: object COUNT is the key causal quantity (one action often
    # adds/removes exactly one object); do not coarsen it away.
    return min(max(n, 0), 15)


def _bin_dist(d) -> int:
    if d is None or d > 8:
        return 3
    if d <= 0:
        return 0
    if d <= 2:
        return 1
    return 2


def _components(mask: np.ndarray):
    """4-connected components of a bool mask.

    Returns list of dicts: size, centroid (cy,cx float), topmost (y,x).
    Pure-python BFS; frames are 64x64 so this is fast enough.
    """
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    # iterate only over set pixels
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        sy = sx = n = 0
        top = (y0, x0)
        while q:
            y, x = q.pop()
            n += 1
            sy += y
            sx += x
            if (y, x) < top:
                top = (y, x)
            if y > 0 and mask[y - 1, x] and not seen[y - 1, x]:
                seen[y - 1, x] = True
                q.append((y - 1, x))
            if y < h - 1 and mask[y + 1, x] and not seen[y + 1, x]:
                seen[y + 1, x] = True
                q.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not seen[y, x - 1]:
                seen[y, x - 1] = True
                q.append((y, x - 1))
            if x < w - 1 and mask[y, x + 1] and not seen[y, x + 1]:
                seen[y, x + 1] = True
                q.append((y, x + 1))
        comps.append(
            {"size": n, "cy": sy / n, "cx": sx / n, "top": top}
        )
    return comps


class FrameParser:
    """Stateful parser: assigns stable color slots in first-seen order."""

    def __init__(self):
        self.slot_of_color = {}  # color value -> slot index 1..N_SLOTS

    def reset_slots(self):
        self.slot_of_color = {}

    def parse(self, frame: np.ndarray):
        frame = np.asarray(frame, dtype=np.int64)
        vals, counts = np.unique(frame, return_counts=True)
        bg = int(vals[int(np.argmax(counts))])

        # non-bg colors sorted by count desc; assign stable slots first-seen
        order = np.argsort(-counts)
        slot_colors = {}  # slot -> color
        for idx in order:
            c = int(vals[idx])
            if c == bg:
                continue
            if c not in self.slot_of_color:
                if len(self.slot_of_color) < N_SLOTS:
                    self.slot_of_color[c] = len(self.slot_of_color) + 1
                else:
                    continue  # more colors than slots: ignore extras
            slot_colors[self.slot_of_color[c]] = c

        ncol = int(np.sum(vals != bg))
        fg = int(np.sum(frame != bg))

        bins = {"ncol": min(ncol, 6), "fg": _bin_count(fg)}
        raw = {"bg": bg, "ncol": ncol, "fg": fg}
        meta = {"slot_color": dict(slot_colors), "bg": bg,
                "top": {}, "centroid": {}, "bg_pixel": None}

        # a background pixel to click for the "null" intervention
        ys, xs = np.nonzero(frame == bg)
        if len(ys):
            # pick one far from non-bg pixels: use the middle of the list
            meta["bg_pixel"] = (int(ys[len(ys) // 2]), int(xs[len(xs) // 2]))

        masks = {}
        biggest = None  # largest foreground component overall (global pos)
        for k in range(1, N_SLOTS + 1):
            c = slot_colors.get(k)
            if c is None:
                bins[f"cnt{k}"] = 0
                bins[f"obj{k}"] = 0
                raw[f"cnt{k}"] = 0
                continue
            mask = frame == c
            masks[k] = mask
            cnt = int(np.sum(mask))
            comps = _components(mask)
            comps.sort(key=lambda d: -d["size"])
            big = comps[0]
            if biggest is None or big["size"] > biggest["size"]:
                biggest = big
            bins[f"cnt{k}"] = _bin_count(cnt)
            bins[f"obj{k}"] = _bin_obj(len(comps))
            raw[f"cnt{k}"] = cnt
            raw[f"obj{k}"] = len(comps)
            meta["top"][k] = big["top"]          # (y, x) click anchor
            meta["centroid"][k] = (big["cy"], big["cx"])

        if biggest is not None:
            bins["gcx"] = int(np.clip(biggest["cx"] // (64 // CX_BINS),
                                      0, CX_BINS - 1))
            bins["gcy"] = int(np.clip(biggest["cy"] // (64 // CX_BINS),
                                      0, CX_BINS - 1))
        else:
            bins["gcx"] = 0
            bins["gcy"] = 0

        # relational: min Manhattan distance between slot 1 and slot 2
        d = None
        if 1 in masks and 2 in masks:
            y1, x1 = np.nonzero(masks[1])
            y2, x2 = np.nonzero(masks[2])
            # subsample for speed
            if len(y1) > 150:
                s = np.linspace(0, len(y1) - 1, 150).astype(int)
                y1, x1 = y1[s], x1[s]
            if len(y2) > 150:
                s = np.linspace(0, len(y2) - 1, 150).astype(int)
                y2, x2 = y2[s], x2[s]
            dd = (np.abs(y1[:, None] - y2[None, :])
                  + np.abs(x1[:, None] - x2[None, :]))
            d = int(np.min(dd))
        bins["d12"] = _bin_dist(d)
        raw["d12"] = -1 if d is None else d

        presence = 0
        for k in slot_colors:
            presence |= 1 << (k - 1)
        meta["presence"] = presence
        return {"bins": bins, "raw": raw, "meta": meta}
