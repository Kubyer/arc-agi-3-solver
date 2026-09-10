"""Planning with the learned program world-models: receding-horizon beam search.

Rolls out the current best program per action key (shallow beam search over
*predicted* frames) and scores trajectories by:
  * novelty      - predicted states unlike anything seen this level
  * ucb          - optimism bonus for under-sampled action keys (drives the
                   falsification loop to gather data where it is thinnest)
  * curiosity    - prediction-error bonus: explore actions whose effects the
                   model still predicts poorly, until they are learned
  * win bonus    - repeating action keys / pre-state patterns that previously
                   incremented levels_completed
  * danger       - avoiding action keys that statistically cause GAME_OVER
  * step cost    - mild preference for shorter plans

ACTION6 (click) branching: clicks are proposed at (a) centroids of small
non-background connected components (likely buttons), (b) cells where clicks
previously changed the frame, (c) unclicked non-background cells, with
regions that already absorbed dead clicks deferred.
"""

import numpy as np

from world_model import frame_hash16, mode_color

DEPTH = 4
WIDTH = 16
N_CLICK_CANDS = 10
STEP_COST = 0.02
WIN_BONUS = 6.0
WIN_KEY_BONUS = 2.0
DANGER_PENALTY = 5.0
NOVELTY_W = 1.0
UCB_W = 1.5
CURIOSITY_W = 2.0
TIE_NOISE = 0.05


class Planner:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.win_memory = []       # (action_key, hash16(pre_frame))
        self.visited = {}          # hash16 -> count, current level
        self.effective_clicks = set()
        self.tried_clicks = set()
        self.dead_regions = {}     # click region key -> # no-change clicks

    # -- level lifecycle ----------------------------------------------------
    def on_level_start(self):
        self.visited = {}
        self.tried_clicks = set()

    def note_win(self, key, pre_hash):
        self.win_memory.append((key, pre_hash))

    def note_frame(self, g32):
        h = frame_hash16(g32)
        self.visited[h] = self.visited.get(h, 0) + 1

    def note_click_effect(self, qx, qy, changed):
        if changed:
            self.effective_clicks.add((qx, qy))
        self.tried_clicks.add((qx, qy))

    def note_click_dead(self, region_key):
        self.dead_regions[region_key] = self.dead_regions.get(region_key, 0) + 1

    def _region_of(self, qx, qy):
        return (6, qx // 4, qy // 4)

    # -- scoring helpers ----------------------------------------------------
    def novelty(self, g32):
        h = frame_hash16(g32)
        return NOVELTY_W / np.sqrt(1.0 + self.visited.get(h, 0))

    def win_bonus(self, key, pre_g32):
        bonus = 0.0
        for k, h in self.win_memory:
            if k != key:
                continue
            bonus = max(bonus, WIN_KEY_BONUS)
            if h == frame_hash16(pre_g32):
                bonus = max(bonus, WIN_BONUS)
        return bonus

    @staticmethod
    def _total_tries(keymodels):
        return sum(m.tries for m in keymodels.values()) + 1

    @staticmethod
    def ucb(keymodels, key, total):
        km = keymodels.get(key)
        n = km.tries if km is not None else 0
        return UCB_W * np.sqrt(np.log(total) / (1.0 + n))

    @staticmethod
    def curiosity(keymodels, key):
        """Prediction-error curiosity: bonus for actions whose effects the
        world-model still predicts poorly. Dead cells (perfectly predicted
        no-ops) get ~0; surprising interactions (buttons, pieces) attract
        exploration until they are learned."""
        km = keymodels.get(key)
        if km is None or km.ema_err is None:
            return CURIOSITY_W * 0.5
        return CURIOSITY_W * np.sqrt(min(km.ema_err, 256.0) / 32.0)

    # -- click proposals ----------------------------------------------------
    def _component_targets(self, g32, bg):
        """Centroids of small non-bg connected components (likely buttons /
        clickables), smallest first. Union-find on the 32x32 grid."""
        h, w = g32.shape
        parent = list(range(h * w))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        fg = g32 != bg
        for y in range(h):
            for x in range(w):
                if not fg[y, x]:
                    continue
                i = y * w + x
                if x + 1 < w and fg[y, x + 1]:
                    union(i, i + 1)
                if y + 1 < h and fg[y + 1, x]:
                    union(i, i + w)
        comps = {}
        for y in range(h):
            for x in range(w):
                if fg[y, x]:
                    r = find(y * w + x)
                    c = comps.get(r)
                    if c is None:
                        comps[r] = [x, y, 1]
                    else:
                        c[0] += x; c[1] += y; c[2] += 1
        out = []
        for sx, sy, n in comps.values():
            if 1 <= n <= 120:
                out.append((sx // n, sy // n, n))
        out.sort(key=lambda t: t[2])
        return [(x, y) for x, y, _ in out]

    def click_candidates(self, g32, bg):
        cands = []
        for qx, qy in self._component_targets(g32, bg):
            if (qx, qy) not in self.tried_clicks and (qx, qy) not in cands:
                cands.append((qx, qy))
            if len(cands) >= N_CLICK_CANDS:
                return cands
        eff = [c for c in self.effective_clicks if c not in self.tried_clicks]
        self.rng.shuffle(eff)
        for qx, qy in eff[:4]:
            cands.append((qx, qy))
        ys, xs = np.nonzero(g32 != bg)
        order = np.arange(len(xs))
        self.rng.shuffle(order)
        deferred = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if (x, y) in self.tried_clicks or (x, y) in cands:
                continue
            # deprioritize regions where clicks already did nothing
            if self.dead_regions.get(self._region_of(x, y), 0) >= 2:
                deferred.append((x, y))
            else:
                cands.append((x, y))
            if len(cands) >= N_CLICK_CANDS:
                break
        if len(cands) < N_CLICK_CANDS:
            cands += deferred[:N_CLICK_CANDS - len(cands)]
        if not cands:
            # degenerate all-background frame: still return grid points
            cands = [(8, 8), (24, 8), (8, 24), (24, 24), (16, 16)]
        return cands[:N_CLICK_CANDS]

    def _branches(self, g32, legal_ids, key_of, n_click_cands=N_CLICK_CANDS):
        bg = mode_color(g32)
        branches = []
        for aid in legal_ids:
            if aid == 6:
                for qx, qy in self.click_candidates(g32, bg)[:n_click_cands]:
                    data = {"x": qx * 2 + 1, "y": qy * 2 + 1}
                    branches.append((aid, data, key_of(aid, data), (qx, qy)))
            else:
                branches.append((aid, None, key_of(aid, None), None))
        return branches

    def _predict(self, keymodels, key, g, click):
        km = keymodels.get(key)
        if km is None:
            return g
        return km.predict(g, click)

    # -- beam search --------------------------------------------------------
    def plan(self, g32, legal_ids, keymodels, key_of):
        """Return (action_id, data) = first step of best predicted trajectory."""
        branches = self._branches(g32, legal_ids, key_of)
        if not branches:
            return None, None
        beam = [(0.0, g32, None)]
        best_first, best_score = 0, float("-inf")
        total = self._total_tries(keymodels)
        for _ in range(DEPTH):
            nxt = []
            for score, g, first in beam:
                for bi, (aid, data, key, click) in enumerate(branches):
                    pred = self._predict(keymodels, key, g, click)
                    r = (self.novelty(pred) - STEP_COST
                         + self.win_bonus(key, g)
                         + self.ucb(keymodels, key, total)
                         + self.curiosity(keymodels, key)
                         + self.rng.normal(0, TIE_NOISE))
                    km = keymodels.get(key)
                    if km is not None and km.dangerous:
                        r -= DANGER_PENALTY
                    ns = score + r
                    f0 = bi if first is None else first
                    nxt.append((ns, pred, f0))
                    if ns > best_score:
                        best_score, best_first = ns, f0
            nxt.sort(key=lambda t: t[0], reverse=True)
            beam = nxt[:WIDTH]
        aid, data, _, _ = branches[best_first]
        return aid, data

    def greedy_novelty_action(self, g32, legal_ids, keymodels, key_of):
        """1-step exploration fallback: maximize predicted novelty + UCB."""
        branches = self._branches(g32, legal_ids, key_of, n_click_cands=4)
        best, best_r = None, float("-inf")
        total = self._total_tries(keymodels)
        for aid, data, key, click in branches:
            pred = self._predict(keymodels, key, g32, click)
            r = (self.novelty(pred) + self.ucb(keymodels, key, total)
                 + self.curiosity(keymodels, key)
                 + self.rng.normal(0, TIE_NOISE))
            km = keymodels.get(key)
            if km is not None and km.dangerous:
                r -= DANGER_PENALTY
            if r > best_r:
                best_r, best = r, (aid, data)
        return best
