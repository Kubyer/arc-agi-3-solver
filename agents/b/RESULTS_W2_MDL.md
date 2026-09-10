# Workstream 2 results: MDL program world-model graft (2026-09-10)

Grafts agent A's MDL program-induction world model (`agents/a/world_model.py`,
loaded by file path — **`agents/a/` was not modified**) into agent B's
object-centric curiosity architecture as a per-action-key forward model.
Design details: `APPROACH.md` §12.

## How to reproduce

```bash
# toy gate (seed 0; benchmark.py always runs seed 0)
.venv/bin/python agents/b/benchmark.py bt11 bt33
# dev subset, same-session comparison (goals on)
MDL_NO_PREDICT=1 MDL_NO_BONUS=1 DEV_TAG=base_final \
    .venv/bin/python agents/b/devbench.py 0 1 2   # baseline (graft disabled)
DEV_TAG=graft_final .venv/bin/python agents/b/devbench.py 0 1 2  # graft
# prediction-error diagnostics (final config; per-seed lines land in
# agents/b/hidden_files/mdl_eval_<tag>.log)
DEV_TAG=eval_final2 .venv/bin/python agents/b/mdl_eval.py 0 1 2
```

Ablation switches (env): `MDL_NO_PREDICT=1` disables all MDL prediction
consumption (score falls back to pure B statistics — behaviorally identical
to the W1 baseline); `MDL_NO_BONUS=1` disables the falsification curiosity
bonus; `MDL_NOOP_PENALTY=1` re-enables the (rejected) grid-exact no-op
penalty.

## Toy gate (non-negotiable, after every change)

Final config, seed 0:

| toy | levels | actions | W-RHAE | gate |
|-----|--------|---------|--------|------|
| bt11 | 5/5 | 72 | 100.0 | ✅ |
| bt33 | 5/5 | 96 | 78.3 | ✅ |

Ablation path that found the final design (bt33, seed 0):

| config | result |
|--------|--------|
| naive MDL-primary signature prediction | 0/5, thousands of actions (broke the gate) |
| + `MDL_NO_BONUS=1` | still broken → bonus not the culprit |
| + `MDL_NO_PREDICT=1` | 96 actions / 78.3 ✅ → prediction consumption was the culprit |
| coarse-novelty without `predicts_noop` penalty (final) | 96 / 78.3 ✅ |

Root cause of the regression: the grid-exact no-op penalty (`predicts_noop`,
≤2 changed cells at 32×32) suppressed an action whose effect is invisible at
32×32 but meaningful in 64px fine space — the 2× subsampling blinds the
program to sub-32×32 effects. The penalty is now behind `MDL_NOOP_PENALTY=1`
(default off).

## Prediction-error diagnostics (dev subset, seeds 0,1,2, final config)

Grid-space mean Hamming cell errors per transition (identity = predict no
change; MDL = program prediction; mature = keys with a stable program,
8+ uses). Fine-signature accuracy = predicted 64px signature == true next
signature (what curiosity's predicted-novelty bonus would need).
Coarse-signature accuracy = same comparison in 32×32 signature space.

| game | identity | MDL | mature | mat/id | fine sig (id/mdl) | coarse sig (id/mdl) |
|------|----------|-----|--------|--------|-------------------|---------------------|
| ar25 | 13.13 | 11.27 | 11.13 | 1.18× | 0.52 / 0.00 | 0.52 / 0.49 |
| g50t | 8.57 | 6.60 | 5.47 | 1.57× | 0.61 / 0.00 | 0.65 / 0.60 |
| lp85 | 6.37 | 6.13 | 5.70 | 1.12× | 0.87 / 0.00 | 0.87 / 0.86 |
| lf52 | 4.40 | 4.20 | 4.50 | 0.98× | 0.65 / 0.00 | 0.65 / 0.63 |
| vc33 | 5.30 | 5.23 | 0.60 | 8.83× | 0.80 / 0.05 | 0.75 / 0.76 |
| mean | 7.55 | 6.69 | 5.48 | **1.38×** | 0.69 / 0.01 | 0.69 / 0.67 |

Reading: mature MDL programs beat identity by 1.38× on graded grid error on
average (range 0.98× on lf52 — non-stationary per-level dynamics defeat the
programs — to 8.83× on vc33). **Coarse-signature accuracy is at identity
parity** (0.67 vs 0.69): the programs get *closer* without matching exactly.
Fine-signature accuracy is ≈ 0.00: **the fine-signature bridge is dead**
(2× subsampling + nearest-neighbor upscaling destroys thin/disconnected
structure; e.g. an ar25 prediction with 12 grid errors vs identity's 54
reparsed into 6 objects vs 12 true objects). No 64px signature is ever
synthesized from a coarse prediction in the final design.

## Dev-subset comparison, same session, seeds 0,1,2 (goals on)

`lvmax` = max levels ever completed (final `levels_completed` returns to 0
after later GAME_OVERs, so `levels_ever` is the honest counter);
`ev` = action index of the completion event. Baseline = graft disabled via
`MDL_NO_PREDICT=1 MDL_NO_BONUS=1` (behaviorally identical to W1 B: verified
by code inspection — with both flags off, `score()`/`do_step()`/`bfs_plan()`
take the exact original paths).

| game | seed | baseline lvmax (ev) | graft lvmax (ev) |
|------|------|---------------------|------------------|
| ar25 | 0 | 0 (–) | 0 (–) |
| ar25 | 1 | 0 (–) | **1 (201)** |
| ar25 | 2 | 0 (–) | 0 (–) |
| g50t | 0–2 | 0 (–) | 0 (–) |
| lp85 | 0 | 1 (9) | 1 (9) |
| lp85 | 1 | 1 (9) | 1 (9) |
| lp85 | 2 | 1 (7) | 1 (7) |
| lf52 | 0 | 1 (22) | 1 (22) |
| lf52 | 1 | 1 (32) | 1 (34) |
| lf52 | 2 | 1 (37) | 1 (37) |
| vc33 | 0–2 | 0 (–) | 0 (–) |

Totals: baseline 6/15 transient L0 completions, graft 7/15. **W-RHAE ≈ 0.0
for both** (graft lp85 seeds 1–2 end at lv=1/8, contributing 0.02–0.03 each
— L0 completed in 196/158 actions vs a 17-action baseline — which rounds to
0.0; every other run ends at lv=0). No completion persists into a
meaningful score.

The only delta is **ar25 seed 1, L0 at action 201** (baseline: no completion
in 1201 actions). All other completions are early (actions 7–37, before any
key reaches the 8-use maturity threshold) and identical or near-identical
between baseline and graft — pre-maturity baseline-curiosity completions the
graft cannot have caused, consistent with the W1 null result.

### Attribution of the ar25 seed-1 completion (levels_ever discipline)

- `from_goal=False`, no test/pursuit in flight: the winning decision was
  curiosity-driven, not goal-directed.
- Winning key `a2`, MDL-mature, falsification bonus 0.00 at the decision.
- Counterfactual re-scoring of every legal candidate at the winning
  decision: the winner ranks #1 under stats-only scoring too
  (graft−stats = −0.034). **The graft shaped the path, not the final
  choice.**
- Trajectory divergence: baseline and graft issue identical keys for 31
  decisions, then diverge at decision 32 (`o5` vs `a4`) — the mature `a4`
  program's coarse-predicted outcome looked already-visited, so the agent
  tried the untried `o5` instead of hammering `a4`. Graft vs `MDL_NO_BONUS=1`
  diverge at decision 41 (`a4` vs `a3`) — the falsification bonus.
- `MDL_NO_BONUS=1` on ar25 seed 1 → no completion: the bonus is load-bearing
  on this trajectory; coarse novelty alone is not sufficient.

### ar25 robustness probe (seeds 3, 4)

| seed | baseline lvmax | graft lvmax (ev) |
|------|---------------|------------------|
| 3 | 0 | **1 (184)**, from_goal=False |
| 4 | 0 | 0 |

ar25 totals: baseline 0/5, graft 2/5 L0 completions (seeds 1, 3; actions 201,
184 — both post-maturity). ar25 is the game where the MDL programs learned
the most meaningful rules (directional displacements: `a1=all(0,-1)`,
`a2/a3/a4` single-axis moves), which is consistent with the coarse-novelty
signal being most informative exactly where the programs are most accurate.

## Compute overhead

- Per learn-step: 32×32 coarsen ≈ 0.7 ms; amortized refit ≈ 0.035 s
  (30 observations incl. refits measured at 0.19 s).
- Wall-clock per game stays within the 180 s budget; graft runs ≈ 10–20%
  slower per game than baseline (dominated by trajectory length differences,
  not per-step cost).

## Verdict

**Partial graft — carry forward with a narrow scope.** What is real:

1. The MDL programs genuinely learn better forward models than identity:
   mature programs beat the identity baseline by **1.38× on graded grid
   error on average** (0.98× on lf52 to 8.83× on vc33; multi-seed; see
   table above).
2. The fine-signature bridge is provably dead (0.00 accuracy) — the final
   design never synthesizes 64px signatures from coarse predictions.
3. The conservative consumption (coarse-space predicted novelty for mature
   keys + falsification curiosity bonus + statistics-only valence) holds the
   toy gate bit-identical and produced 2 transient ar25 L0 completions in 5
   seeds vs 0 for baseline, with the causal path attributed (not the final
   decision, not the goal module).

What is not real: any persistent score improvement — W-RHAE ≈ 0.0 for both
variants on the dev subset (two graft lp85 runs contribute 0.02–0.03 each),
and no 27-game run is warranted.

Recommended carry-forward: keep the MDL world model as a passive forward
model + falsification signal and the two conservative consumption channels;
do not pursue fine-signature prediction from subsampled grids without a
topology-preserving representation; the grid-exact no-op penalty direction
is dead (it blinds the agent to sub-32×32 effects).
