# sweep.py
from baselines import evaluate, restore_alerts

for p in (0.05, 0.1, 0.2, 0.3):
    row = []
    for topo in ("chain", "ring", "star", "bridge"):
        gt, _, _ = evaluate(restore_alerts, topo, episodes=40, p_spread=p)
        row.append(f"{topo}={gt:6.1f}")
    print(f"p_spread={p:4.2f}  " + "  ".join(row))