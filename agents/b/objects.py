"""Object parsing for 64x64 ARC-AGI-3 frames.

Core-knowledge-flavored inductive bias: the world is made of discrete objects
(colored connected components), not pixels. Every downstream component
(transition model, archive, planner) operates on object lists / signatures.
"""

import numpy as np
from scipy import ndimage

_FOUR_CONN = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=int)

CELL = 8  # spatial coarsening for signatures: 64px -> 8 cells per dim


def size_bin(n):
    """Coarse size bucket keeps signatures compact."""
    if n <= 1:
        return 0
    if n <= 4:
        return 1
    if n <= 16:
        return 2
    if n <= 64:
        return 3
    return 4


def signature(objs):
    """Canonical hashable descriptor of an object-relational state.

    Sorted tuple of (color, coarse_y, coarse_x, size_bin) per object.
    Coarsening keeps the Go-Explore archive small while preserving the
    relational layout that matters for these games.
    """
    return tuple(sorted(
        (o["color"], int(o["cy"]) // CELL, int(o["cx"]) // CELL, o["size_bin"])
        for o in objs
    ))


def parse_frame(frame):
    """Parse a (64,64) int8 frame into a list of object dicts.

    Background = most frequent color (decided empirically; beats assuming
    color 0, since several games use 0 as a sprite color and others use
    e.g. 5 as background). Objects = 4-connected components per color.

    Returns (objs, sig, bg) where each obj has:
      color, size, size_bin, cy, cx, bbox
    """
    frame = np.asarray(frame)
    vals, counts = np.unique(frame, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])

    objs = []
    for v in vals:
        v = int(v)
        if v == bg:
            continue
        labelled, n = ndimage.label(frame == v, structure=_FOUR_CONN)
        if n == 0:
            continue
        sizes = np.bincount(labelled.ravel(), minlength=n + 1)[1:]
        for i in range(1, n + 1):
            sz = int(sizes[i - 1])
            ys, xs = np.nonzero(labelled == i)
            cy = float(ys.mean())
            cx = float(xs.mean())
            objs.append({
                "color": v,
                "size": sz,
                "size_bin": size_bin(sz),
                "cy": cy,
                "cx": cx,
                "bbox": (int(ys.min()), int(xs.min()),
                         int(ys.max()), int(xs.max())),
            })
    objs.sort(key=lambda o: (o["color"], o["size_bin"],
                             round(o["cy"], 1), round(o["cx"], 1)))
    return objs, signature(objs), bg


def colors_of(objs):
    return {o["color"] for o in objs}
