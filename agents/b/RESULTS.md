# Builder B benchmark results

Agent: object-centric world model + curiosity exploration (Go-Explore). Offline, CPU-only, numpy/scipy. Seed 0, max 4000 counted actions / 180s wall per game.

Per-level RHAE = min(((baseline/actions)^2)*100, 115); W-RHAE = level-index-weighted mean, 1-based weights (official formula).

```
bt11   tag1,tag2 lv=5/5 act=72 go=0 ret=0 arch=68 wall=0s W-RHAE=100.0 | L0:4/4=100.0 L1:8/8=100.0 L2:16/16=100.0 L3:20/20=100.0 L4:24/24=100.0
bt33   tag3,tag4 lv=5/5 act=96 go=2 ret=0 arch=72 wall=1s W-RHAE=78.3 | L0:9/10=115.0 L1:9/15=115.0 L2:33/20=36.7 L3:20/20=100.0 L4:25/20=64.0
```

MEAN W-RHAE: 89.2 | total levels completed: 10
