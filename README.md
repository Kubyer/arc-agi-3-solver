# ARC-AGI-3 Solver — Self-Falsifying World Models + Object-Centric Curiosity

Entry for the **ARC Prize 2026** (ARC-AGI-3 track + Paper Track) by [@Kubyer](https://github.com/Kubyer).

ARC-AGI-3 poses interactive, instruction-free reasoning environments: the agent
sees only 64×64 pixel frames and must discover the controls, the dynamics, and
the win condition by acting. Scoring rewards human-like action efficiency. This
repository contains three fully offline, CPU-friendly symbolic agents built to
attack that problem — no LLMs, no internet, no learned priors over games.

## The headline finding

Across three independently-designed agents and five follow-up experiments, the
same wall appeared every time: **dynamics learning works, goal discovery is the
binding constraint.** Every agent learns *how* its environment behaves, but none
can reliably infer *what counts as done* — and curiosity alone never discovers
the goal. That convergent result is the empirical core of our prize paper.

## Agents

| Agent | Idea | Status |
|---|---|---|
| [`agents/a`](agents/a) | **Self-falsifying program world-models (MDL search).** Hypothesizes small Python programs predicting transitions, keeps those minimizing `S = E + λL` (prediction error + description length), refits on falsification. | Supporting experiment; theory core of the paper |
| [`agents/b`](agents/b) | **Object-centric world model + curiosity.** Parses frames into objects, learns per-action transition statistics, explores with a Go-Explore archive + novelty bonus, plans with beam search. Grafted with A's MDL forward models and falsification bonus. | **Lead agent** |
| [`agents/c`](agents/c) | **Causal world model via interventions.** Treats actions as interventions in a structural causal model over object variables; plans by simulating interventions. | Supporting experiment |

All three solve the toy environments reproducibly (`bt11` 5/5 @ 100.0,
`bt33` 5/5 @ 78.3). On the 25 public ARC-AGI-3 games they learn correct local
dynamics but stall on goal inference — see each agent's `RESULTS.md` for honest,
multi-seed numbers, including retracted misattributions.

## Repository layout

```
├── README.md
├── LICENSE                  # CC-BY-4.0 (per competition winner requirements)
├── ENV_API.md               # Exact ARC-AGI-3 environment API reference
├── agents/
│   ├── a/                   # MDL program world-models
│   ├── b/                   # Object-centric curiosity (lead agent)
│   └── c/                   # Causal world model
```

Each agent directory holds runnable code, an `APPROACH.md` design document, and
`RESULTS.md` benchmark tables. Run any agent's benchmark with:

```
python agents/<a|b|c>/benchmark.py
```

(requires the `arc-agi` + `arcengine` packages and ARC-AGI-3 environment files)

## Reproducibility

Every claimed number in this repository comes from ≥3 seeded runs with paired
same-session baselines. Single-seed wins that did not reproduce were retracted
in the docs, not hidden. Toy baselines (`bt11`, `bt33`) are regression-gated
bit-identical across all iterations.

## License

CC-BY 4.0 — see [LICENSE](LICENSE).
