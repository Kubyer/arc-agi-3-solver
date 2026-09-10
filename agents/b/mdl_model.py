"""MDL program world-model graft for agent B (workstream 2).

Wraps agent A's KeyModel/Rule machinery (agents/a/world_model.py -- loaded
by file path, agents/a is never modified) behind B's action-key space:

  B key "a1".."a5" -> one KeyModel per keyboard action, click=None
  B key "o<C>"     -> one KeyModel per clicked color C; the *actual*
                      resolved click coordinates are recorded at execution
                      and passed through, so the `goto` mode can learn
                      "move the foreground centroid to the click position"
  B key "g0".."g3" -> one KeyModel per fixed grid-click position
  global fallback  -> one KeyModel shared by all click keys, like A's
                      (6, "any"): predicts for click keys with no data yet

Observation: every learn=True transition feeds
(coarsen(frame_before), coarsen(frame_after), click32|None) into the key's
KeyModel (plus the shared click fallback for click keys).
Falsification-triggered refits come free from KeyModel.observe.

Prediction: predict_grid(key, grid32, click64) -> the key's program's
predicted next 32x32 grid. Consumption happens in this NATIVE coarse /
grid space: coarse_sig32() (predicted coarse novelty) and
predicts_noop() (grid-exact no-op detection). No 64px signature is ever
synthesized from a coarse prediction -- that bridge is provably lossy
(2x subsampling destroys thin/disconnected structure; see
RESULTS_W2_MDL.md). Mature = enough uses + a rule + no recent
falsification; immature keys fall back to B's statistics table
(augment-first, never rip-and-replace).

A falsified rule (prediction error > max(12, 2xEMA), the same trigger that
fires a refit) is exposed as a principled surprise/learning signal: it
feeds a small prediction-error curiosity bonus for the recently-falsified
key (A's own curiosity idea), and the falsification count is reported for
diagnostics. Raw falsifications do NOT trigger B's early-reset/tabu --
level changes legitimately falsify rules, so that would punish winning
trajectories.

Diagnostics: record_eval()/eval_summary() measure MDL vs identity vs
statistics prediction error (grid Hamming + signature accuracy) --
behavior-neutral, enabled with MDL_EVAL=1. Ablation switches (env):
MDL_NO_PREDICT=1 disables all prediction consumption (baseline
curiosity); MDL_NO_BONUS=1 disables the falsification bonus.

Numpy only. Refit cost is ~0.03-0.08s (A's vectorized delta fit).
"""

import importlib.util
import os
import zlib

import numpy as np

from .model import TransitionModel
from .objects import parse_frame, signature, size_bin

GRID32 = 32


def _load_a_world_model():
    """Import agents/a/world_model.py by file path (agents/a untouched)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "a", "world_model.py"))
    spec = importlib.util.spec_from_file_location("a_world_model_graft", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_WM = _load_a_world_model()


def coarsen_frame(frame64):
    """(64,64) -> (32,32) int8, same as A's coarsen."""
    return _WM.coarsen(np.ascontiguousarray(
        np.asarray(frame64, dtype=np.int8)))


def render_objs32(objs, bg):
    """Render an object list onto a 32x32 grid (for imagined states).

    Objects are drawn as filled bbox rectangles (small objects last, so
    sprites draw over background-ish rects); objects with a degenerate
    bbox (synthesized appearances) are stamped as 3x3 cells at the
    centroid. Used when no real frame is available (planning rollouts).
    """
    g = np.full((GRID32, GRID32), int(bg) & 15, dtype=np.int8)
    for o in sorted(objs, key=lambda o: o["size"]):
        c = int(o["color"]) & 15
        y0, x0, y1, x1 = o["bbox"]
        if x1 <= x0 and y1 <= y0:
            cx32 = int(np.clip(o["cx"] // 2, 0, GRID32 - 1))
            cy32 = int(np.clip(o["cy"] // 2, 0, GRID32 - 1))
            g[max(0, cy32 - 1):cy32 + 2, max(0, cx32 - 1):cx32 + 2] = c
            continue
        xs0 = int(np.clip(x0 // 2, 0, GRID32 - 1))
        xs1 = int(np.clip(x1 // 2, 0, GRID32 - 1))
        ys0 = int(np.clip(y0 // 2, 0, GRID32 - 1))
        ys1 = int(np.clip(y1 // 2, 0, GRID32 - 1))
        g[ys0:ys1 + 1, xs0:xs1 + 1] = c
    return g


def coarse_sig32(grid32):
    """Signature of a 32x32 grid, mapped into B's 64px signature space.

    Objects parsed at 32x32 have positions x2 and sizes x4 so the bins
    (8px spatial, log size) match objects.signature's granularity. This
    is the representation in which MDL predicted-novelty is judged:
    unlike upscaling the predicted grid to 64x64 and re-parsing (which
    can never exactly reproduce a true 64px signature -- <=4px objects
    are lost in 2x subsampling), the coarse signature is computed the
    same way for observed and predicted coarse grids, so the comparison
    is apples-to-apples.
    """
    objs, _, _ = parse_frame(grid32)
    return tuple(sorted(
        (o["color"], int(o["cy"] * 2) // 8, int(o["cx"] * 2) // 8,
         size_bin(o["size"] * 4))
        for o in objs))
    return zlib.crc32(("%d|%s" % (seed, b_key)).encode("utf-8"))


def _stable_seed(seed, b_key):
    return zlib.crc32(("%d|%s" % (seed, b_key)).encode("utf-8"))


class MDLWorldModel:
    """Per-B-action-key MDL program models."""

    # Minimum observations before a key's program is trusted for
    # prediction. Below this, observe() still learns passively but
    # predict_forward() returns None (statistics fallback).
    MATURE_USES = 8
    # A program is only trusted after it has survived this many recent
    # uses without falsification: churning programs (dynamics not yet
    # captured) never drive behavior, stable ones do.
    STABLE_WINDOW = 10
    # Steps after a falsification during which the key gets a curiosity
    # bonus (prediction-error curiosity: probe what we just got wrong).
    FALS_BONUS_WINDOW = 6
    FALS_BONUS = 0.6

    def __init__(self, seed=0, lam=5.0):
        self.seed = seed
        self.lam = lam
        self.keys = {}  # b_key -> KeyModel
        self.click_fallback = _WM.KeyModel(
            lam=lam, seed=_stable_seed(seed, "click_fallback"))
        self.last_falsified = {}  # b_key -> bool from most recent observe()
        self.last_fals_uses = {}  # b_key -> key.uses at last falsification
        self.n_falsified = 0
        self.eval = None  # enabled by enable_eval(): prediction-error stats
        # Coarse-signature archive: visit counts of observed 32x32 states,
        # in which the MDL programs' predicted novelty is judged
        # (apples-to-apples; see coarse_sig32).
        self.coarse_visits = {}
        # Ablation switches (env): isolate consumption paths.
        self.use_predict = os.environ.get("MDL_NO_PREDICT") != "1"
        self.use_bonus = os.environ.get("MDL_NO_BONUS") != "1"

    # -- key management ------------------------------------------------

    @staticmethod
    def is_click_key(b_key):
        return b_key[0] in ("o", "g")

    def key_model(self, b_key):
        km = self.keys.get(b_key)
        if km is None:
            fb = self.click_fallback if self.is_click_key(b_key) else None
            km = _WM.KeyModel(lam=self.lam,
                              seed=_stable_seed(self.seed, b_key),
                              fallback=fb)
            self.keys[b_key] = km
        return km

    def note_gameover(self, b_key):
        km = self.key_model(b_key)
        km.note_gameover()
        if self.is_click_key(b_key):
            self.click_fallback.note_gameover()

    # -- observation ----------------------------------------------------

    def observe(self, b_key, frame64_before, frame64_after, click64=None):
        """Feed one transition. click64 = (x, y) in 64-px, or None."""
        km = self.key_model(b_key)
        s = coarsen_frame(frame64_before)
        s2 = coarsen_frame(frame64_after)
        click32 = ((click64[0] // 2, click64[1] // 2)
                   if click64 is not None else None)
        # Falsification check, mirroring KeyModel.observe's own trigger
        # (err computed on the pre-update rule; ema as updated by
        # note_prediction). Computed here so the signal is available to
        # the agent without re-running prediction.
        falsified = False
        if km.rule is not None:
            err = int(np.count_nonzero(km.rule.predict(s, click32) != s2))
            ema = err if km.ema_err is None \
                else 0.9 * km.ema_err + 0.1 * err
            falsified = err > max(12.0, 2.0 * ema)
        km.observe(s, s2, click32)
        if self.is_click_key(b_key):
            self.click_fallback.observe(s, s2, click32)
        self.last_falsified[b_key] = falsified
        if falsified:
            self.last_fals_uses[b_key] = km.uses
            self.n_falsified += 1

    # -- prediction ------------------------------------------------------

    def effective_rule(self, b_key):
        km = self.keys.get(b_key)
        if km is None:
            return None
        if km.rule is not None:
            return km.rule
        return km.fallback.rule if km.fallback is not None else None

    def mature(self, b_key):
        """A program worth trusting for prediction exists for this key:
        enough observations, a rule, and no recent falsification."""
        if not self.use_predict:
            return False
        km = self.keys.get(b_key)
        if km is None or km.uses < self.MATURE_USES:
            return False
        if self.effective_rule(b_key) is None:
            return False
        lf = self.last_fals_uses.get(b_key)
        if lf is not None and km.uses - lf <= self.STABLE_WINDOW:
            return False
        return True

    def predict_grid(self, b_key, grid32, click64=None):
        """grid32 -> predicted grid32, or None when immature."""
        if not self.mature(b_key):
            return None
        km = self.keys[b_key]
        click32 = ((click64[0] // 2, click64[1] // 2)
                   if click64 is not None else None)
        return km.predict(grid32, click32)

    def curiosity_bonus(self, b_key):
        """Prediction-error curiosity: recently falsified keys attract
        exploration (A's idea), without touching reset/tabu logic."""
        if not self.use_bonus:
            return 0.0
        km = self.keys.get(b_key)
        if km is None or km.uses < self.MATURE_USES:
            return 0.0
        lf = self.last_fals_uses.get(b_key)
        if lf is not None and km.uses - lf <= self.FALS_BONUS_WINDOW:
            return self.FALS_BONUS
        return 0.0

    # -- coarse-space novelty -------------------------------------------

    def note_coarse(self, csig):
        """Register one observed coarse state (called per main-loop step)."""
        self.coarse_visits[csig] = self.coarse_visits.get(csig, 0) + 1

    def coarse_psig(self, b_key, grid32, click64=None):
        """Predicted coarse signature for a mature key, or None."""
        pred = self.predict_grid(b_key, grid32, click64)
        if pred is None:
            return None
        return coarse_sig32(pred)

    def predicts_noop(self, b_key, grid32, click64=None):
        """Grid-exact no-op judgment for a mature key (None if immature).

        Sharper than the statistics' signature-based no-op rate: a key
        whose program predicts <=2 changed cells is a true no-op even
        when sub-signature pixel changes occur (and vice versa).
        """
        pred = self.predict_grid(b_key, grid32, click64)
        if pred is None:
            return None
        return int(np.count_nonzero(pred != grid32)) <= 2

    # -- prediction-error evaluation (diagnostic, behavior-neutral) ------

    def enable_eval(self):
        self.eval = {"n": 0, "identity": 0,
                     "mdl": 0, "mdl_n": 0, "mdl_mat": 0, "mdl_mat_n": 0,
                     "stats": 0, "stats_n": 0,
                     "sig_id": 0, "sig_mdl": 0, "sig_mdl_n": 0,
                     "sig_stats": 0, "sig_stats_n": 0,
                     "sig_cid": 0, "sig_cmdl": 0, "sig_cmdl_n": 0}

    def record_eval(self, b_key, f32b, f32a, objs_before, objs_after,
                    click64, stats_predict_fn):
        """Prediction-error comparison (diagnostic, behavior-neutral).

        Grid space: Hamming cell errors of identity / MDL-program /
        rendered-B-stats predictions vs the true next 32x32 frame.
        Signature space: whether the predicted next signature equals the
        true next signature (this is what curiosity's predicted-novelty
        bonus actually consumes). Pure functions only: no RNG, no state
        mutation.
        """
        e = self.eval
        if e is None:
            return
        e["n"] += 1
        e["identity"] += int(np.count_nonzero(f32b != f32a))
        true_sig = signature(objs_after)
        e["sig_id"] += int(signature(objs_before) == true_sig)
        true_csig = coarse_sig32(f32a)
        e["sig_cid"] += int(coarse_sig32(f32b) == true_csig)
        km = self.keys.get(b_key)
        rule = self.effective_rule(b_key)
        if km is not None and rule is not None:
            click32 = ((click64[0] // 2, click64[1] // 2)
                       if click64 is not None else None)
            pred = km.predict(f32b, click32)
            e["mdl"] += int(np.count_nonzero(pred != f32a))
            e["mdl_n"] += 1
            e["sig_cmdl"] += int(coarse_sig32(pred) == true_csig)
            e["sig_cmdl_n"] += 1
            up = pred.repeat(2, axis=0).repeat(2, axis=1)
            _po, psig, _bg = parse_frame(up)
            e["sig_mdl"] += int(psig == true_sig)
            e["sig_mdl_n"] += 1
            if self.mature(b_key):
                e["mdl_mat"] += int(np.count_nonzero(pred != f32a))
                e["mdl_mat_n"] += 1
        so, _added = stats_predict_fn(b_key, objs_before)
        if so is not None:
            g = render_objs32(so, rule.bg if rule is not None else 0)
            e["stats"] += int(np.count_nonzero(g != f32a))
            e["stats_n"] += 1
            e["sig_stats"] += int(signature(so) == true_sig)
            e["sig_stats_n"] += 1

    def eval_summary(self):
        e = self.eval
        if e is None or e["n"] == 0:
            return None
        out = {"n": e["n"], "identity_mean": e["identity"] / e["n"],
               "sig_acc_identity": e["sig_id"] / e["n"],
               "sig_cacc_identity": e["sig_cid"] / e["n"]}
        if e["mdl_n"]:
            out["mdl_mean"] = e["mdl"] / e["mdl_n"]
            out["mdl_n"] = e["mdl_n"]
            out["mdl_vs_identity"] = out["identity_mean"] / max(
                1e-9, out["mdl_mean"])
            out["sig_acc_mdl"] = e["sig_mdl"] / e["sig_mdl_n"]
            out["sig_cacc_mdl"] = e["sig_cmdl"] / e["sig_cmdl_n"]
        if e["mdl_mat_n"]:
            out["mdl_mature_mean"] = e["mdl_mat"] / e["mdl_mat_n"]
            out["mdl_mature_n"] = e["mdl_mat_n"]
        if e["stats_n"]:
            out["stats_mean"] = e["stats"] / e["stats_n"]
            out["stats_n"] = e["stats_n"]
            out["sig_acc_stats"] = e["sig_stats"] / e["sig_stats_n"]
        return out

    # -- reporting --------------------------------------------------------

    def summary(self):
        rules = {}
        for k, km in self.keys.items():
            r = self.effective_rule(k)
            if r is not None:
                ema = km.ema_err
                rules[k] = "%s uses=%d ema=%.1f refits=%d" % (
                    repr(r), km.uses, ema if ema is not None else -1,
                    km.refits)
        return {"keys": len(self.keys), "mature": sum(1 for k in self.keys
                                                     if self.mature(k)),
                "n_falsified": self.n_falsified,
                "click_fallback_refits": self.click_fallback.refits,
                "click_fallback_uses": self.click_fallback.uses,
                "rules": rules}
