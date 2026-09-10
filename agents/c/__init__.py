"""Builder C: causal world model via interventions.

Modules:
  variables - frame -> 17 binned causal variables
  causal    - interventional data -> invariance-tested causal graph
  planner   - win-direction + greedy SCM simulation + beam search
  agent     - episode loop; run_episode(env, game_id, baselines, ...) entry
  benchmark - official-style RHAE benchmark over toy + real games
"""
from agent import CausalAgent, run_episode

__all__ = ["CausalAgent", "run_episode"]
