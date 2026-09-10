"""Go-Explore style archive over object-relational states.

Cells are keyed by object signature (see objects.signature). Each cell
stores visit counts, the first trajectory that reached it from the level
start (used for cheap returns: RESET is score-free, so replaying a stored
trajectory from the level start is the return mechanism), the legal
action-keys observed there, and which keys have been tried.

Frontier selection: least-visited cell that still has untried,
non-doomed action-keys.
"""

import random


class Archive:
    def __init__(self):
        self.cells = {}

    def __len__(self):
        return len(self.cells)

    def visit(self, sig, legal_keys):
        """Record a visit. Returns True if this cell is new."""
        c = self.cells.get(sig)
        if c is None:
            self.cells[sig] = {"visits": 1, "traj": None,
                               "tried": set(), "legal": set(legal_keys)}
            return True
        c["visits"] += 1
        c["legal"] |= set(legal_keys)
        return False

    def set_traj(self, sig, traj):
        c = self.cells.get(sig)
        if c is not None and c["traj"] is None:
            c["traj"] = list(traj)

    def mark_tried(self, sig, key):
        c = self.cells.get(sig)
        if c is not None:
            c["tried"].add(key)

    def tried(self, sig):
        c = self.cells.get(sig)
        return c["tried"] if c else set()

    def frontier(self, is_doomed=None):
        """Least-visited cell with an untried key and a trajectory."""
        if is_doomed is None:
            is_doomed = lambda k: False
        best = None
        for sig, c in self.cells.items():
            if c["traj"] is None:
                continue
            untried = [k for k in c["legal"]
                       if k not in c["tried"] and not is_doomed(k)]
            if not untried:
                continue
            score = (c["visits"], len(c["traj"]))
            if best is None or score < best[0]:
                best = (score, sig)
        return best[1] if best else None

    def frontier_sigs(self, is_doomed=None, max_visits=3, limit=8):
        """Candidate goal signatures for model-based planning."""
        if is_doomed is None:
            is_doomed = lambda k: False
        out = []
        for sig, c in self.cells.items():
            if c["visits"] > max_visits:
                continue
            if any(k not in c["tried"] and not is_doomed(k)
                   for k in c["legal"]):
                out.append(sig)
            if len(out) >= limit:
                break
        return out
