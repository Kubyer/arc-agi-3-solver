# Round 2 iteration results — planner strengthening (W1) + MDL graft (W2)

Agent B (object-centric + curiosity), 2026-09-10. Both workstreams measured
against the same-session B baseline with ≥3 seeds. Toy regression gate:
bt11 5/5 @ 100.0 (exactly 72 actions), bt33 5/5 @ 78.3 — exact in both
workstreams, before and after.

Full detail: `RESULTS_W1_PLANNER.md`, `RESULTS_W2_MDL.md`,
`APPROACH.md` §11 (W1), §12 (W2).

## Per-workstream results (multi-seed)

### W1 — planner strengthening: honest null

Tried, in order: distributional transitions (`sample_objects`), deeper
heuristic search (`plan_beam`, depth 12, MC-validated), pixel-level precision
(`fine_hash`, 2px), goal-directed behavior from frame 1 + sustained pursuit,
3 hypotheses per evaluation, manipulability filtering, hindsight credit.

Dev subset (ar25, g50t, lp85, lf52, vc33), seeds 0–2, goals on vs `GOALS_OFF=1`:

| metric | baseline | W1 candidate |
|---|---|---|
| robust levels solved | 0/15 | 0/15 |
| mean W-RHAE | 0.0 | 0.0 |
| lf52 wall time | 29.7±2.0 s | 97.0±9.5 s (overhead only) |

Key correction: an apparent lf52 L0 solve was **misattributed** to the new
planner. Paired probes (seeds 0,1,2, goals on vs off) showed completions at
*identical action indices* (22, 32) with the planner on or off — before any
goal action was ever taken. The claim was retracted in §11 and
`RESULTS_W1_PLANNER.md`. With zero bundle effect, per-direction ablations
were deliberately not run (nothing to attribute).

### W2 — MDL program graft: partial, narrow carry-forward

Design: A's `KeyModel` (Rule = MODE + sparse DELTA, S = E + λL, λ=5.0,
falsification-driven refits) loaded per B action key on 32×32 frames
(`mdl_model.py`; `agents/a/` untouched). Two aggressive consumptions were
rejected (fine-signature synthesis from coarse grids broke bt33; grid-exact
no-op penalty suppressed sub-32×32 effects). Surviving design is
conservative: B's statistics keep *all* prediction duties; MDL contributes
only (a) predicted-novelty in coarse signature space for mature keys
(8+ uses), and (b) a bounded falsification curiosity bonus. Ablation
switches: `MDL_NO_PREDICT=1`, `MDL_NO_BONUS=1`, `MDL_NOOP_PENALTY=1`.

Prediction quality (final config, seeds 0–2): mature MDL beats identity
**1.38× on grid Hamming on average** — 8.83× vc33, 1.57× g50t, 1.1–1.2×
ar25/lp85, 0.98× lf52 (non-stationary). Coarse-signature accuracy at parity
with B's statistics (0.67 vs 0.69); **fine-signature synthesis ≈ 0.00 —
that bridge is dead**.

Game results (dev subset, seeds 0–2, same session): baseline 6/15 vs graft
7/15 transient L0 completions; the only delta is ar25 seed 1 L0 @ action
201. Extended ar25 (seeds 3–4): baseline 0/2, graft 1/2 (seed 3 @184) →
**2/5 vs 0/5 on ar25, transient only**. W-RHAE ≈ 0.0 both. Attribution via
counterfactual re-scoring: the ar25 win was curiosity-driven, but the graft
**shaped the path** — trajectories diverge at decision #32 on coarse
novelty and #41 on the falsification bonus; `MDL_NO_BONUS=1` kills the
completion, so the bonus is load-bearing even though the final winning
action ranks #1 under stats-only scoring too.

## What worked
- MDL forward models genuinely predict better than identity (1.38× avg,
  up to 8.83×) at negligible CPU cost — the theory half of the paper is
  further validated inside B's harness.
- The falsification bonus is the first mechanism in two rounds with a
  **causally attributed** (ablation-verified) exploration effect: 2/5 vs
  0/5 transient ar25 L0 completions.
- Measurement discipline held: paired same-session baselines, ≥3 seeds,
  attribution instrumentation (`levels_ever` + winning-decision capture),
  one retracted misattribution, toy gates bit-identical throughout.

## What failed, and why
- **No persistent score change from either workstream.** W1: 0.0 delta on
  every metric. W2: transient L0 completions only, W-RHAE ≈ 0.0 both —
  no 27-game run was warranted by either gate.
- Stronger planning (depth-12 beam, distributional rollouts, fine-grained
  precision) cannot convert hypotheses into solves: real-game plans fail on
  unmodeled walls, click timing, and obstacle geometry the per-key models
  don't represent. The planner optimizes hand-written predicate progress,
  not a learned success signal.
- The eight hand-written goal templates don't cover all game types (g50t:
  zero detected outlines → goal module is a no-op); wrong-hypothesis
  pursuits burn actions without solving.
- Fine signatures cannot be synthesized from coarse-grid MDL predictions —
  the coarsening that makes program search tractable destroys the precision
  planning needs. Unresolved tension (same as §7).

## Convergent finding (now 5 experiments deep)
Three base agents + goal-hypothesis module + planner strengthening + MDL
graft all hit the same wall: **dynamics/world-model learning works in
every paradigm tried; goal discovery is the binding constraint.** Agents
learn *how* the world works but cannot infer *what counts as done* or
compose long unrewarded sequences. This is the paper's headline empirical
result, now replicated across five independent interventions.

## Recommended final configuration
`code/agents/b/` at W2 final state:
- `mdl_model.py` — per-key MDL forward models (passive predictor +
  falsification signal only)
- `agent.py` — curiosity policy + W1 attribution instrumentation + MDL hooks
- `model.py`, `goals.py`, `archive.py`, `objects.py`, `mdl_eval.py`,
  `benchmark.py`, `devbench.py`, `probe_lf52.py`
- Docs: `APPROACH.md` (§1–12), `RESULTS.md`, `RESULTS_W1_PLANNER.md`,
  `RESULTS_W2_MDL.md`, this file
- `agents/a/` and `agents/c/` retained as supporting experiments for the paper.

Toy gate at final state (seed 0): bt11 5/5, 72 actions, 100.0 ✓;
bt33 5/5, 96 actions, 78.3 ✓ — bit-identical to pre-round-2.

Caveats: W1's goal-pursuit machinery is retained but unproven (adds wall
overhead, ~3× on lf52); the only causally-attributed delta in the bundle is
the MDL falsification bonus. Real-game numbers remain single-episode and
transient — multi-seed evaluation is required before any paper claim beyond
the toys.

## Single most important next step
**Replace the hand-written goal-predicate templates with a learned
success signal.** Both rounds show the planner is strong enough to pursue a
target but has nothing worth pursuing: learn, from the thousands of
trajectories the archive already collects, which object-relational features
predict eventual level completion (a success classifier / progress measure
trained on the sparse `levels_completed` signal, transferred across levels
via template priors). Everything else — deeper search, better models —
amplifies a target; only a learned target changes what the agent aims at.
