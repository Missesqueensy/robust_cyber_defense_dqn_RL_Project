# reward.py
COSTS = {"noop": 0.0, "isolate": 0.5, "restore": 1.0, "patch": 0.1}


def compute_reward(n_comp: int, n_total: int, kind: str) -> float:
    """R_t = 1[Ncomp=0] + 0.1*Nsains - 1.0*Ncomp - coût de l'action (état réel, après l'action)."""
    n_sains = n_total - n_comp
    bonus = 1.0 if n_comp == 0 else 0.0
    return bonus + 0.1 * n_sains - 1.0 * n_comp - COSTS[kind]