"""Diagnostics: world-model prediction accuracy per action key.

Runs the agent briefly, then for each learned key reports buffer size,
rule description, and mean cell-error of the rule on its own buffer
(vs the identity baseline). Used to tune lambda.
"""
import os, sys
import numpy as np
from numpy import nan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_agi import Arcade, OperationMode
from mdl_agent import MDLAgent

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

def main(games, n_actions=120, lam=5.0):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(CODE, "environment_files"))
    for gid in games:
        env = client.make(game_id=gid, seed=0)
        agent = MDLAgent(lam=lam, seed=0, max_actions=n_actions,
                         time_budget=240.0)
        stats = agent.run_episode(env)
        print(f"== {gid}: levels={stats['levels_completed']} "
              f"actions={stats['actions']} ==")
        for key in sorted(agent.keymodels, key=str):
            km = agent.keymodels[key]
            pairs = list(km.buf)
            if not pairs:
                continue
            ident_err = np.mean([np.count_nonzero(a != b) for a, b, _ in pairs])
            rule_err = (np.mean([np.count_nonzero(km.rule.predict(a, c) != b)
                                 for a, b, c in pairs])
                        if km.rule is not None else float("nan"))
            print(f"  key={key} n={len(pairs)} rule={km.rule} "
                  f"ident_err={ident_err:.1f} rule_err={rule_err:.1f} "
                  f"danger={km.dangerous} refits={km.refits}")
        ca = agent.click_any
        if len(ca.buf):
            pairs = list(ca.buf)
            ident_err = np.mean([np.count_nonzero(a != b) for a, b, _ in pairs])
            rule_err = np.mean([np.count_nonzero(ca.rule.predict(a, c) != b)
                                for a, b, c in pairs]) if ca.rule else nan
            print(f"  click_any n={len(pairs)} rule={ca.rule} "
                  f"ident_err={ident_err:.1f} rule_err={rule_err:.1f}")

if __name__ == "__main__":
    games = sys.argv[1:] or ["ar25", "lf52", "g50t"]
    lam = float(os.environ.get("LAM", "5.0"))
    main(games, lam=lam)
