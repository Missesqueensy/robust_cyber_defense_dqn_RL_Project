# baselines.py : politiques de référence + évaluation avec IC bootstrap
import csv
import random
import numpy as np

from make_env import make_env
from defense_env import DefenseEnv

N = 10
NOOP = 3 * N                      # indice 30 : ne rien faire
MAX_STEPS = 100
TOPOS = ["chain", "ring", "star", "bridge"]
SEED0 = 1000                      # graines d'évaluation : 1000..1099
EPISODES = 100


# ---------- politiques ----------
def random_policy(obs):
    return random.randrange(3 * N + 1)

def noop_policy(obs):
    return NOOP

def restore_alerts(obs):
    """Restaure un hôte détecté compromis, sinon un hôte en alerte, sinon No-Op."""
    last = obs[-3 * N:]                       # trame la plus récente
    known, alerts = last[:N], last[N:2 * N]
    cand = [i for i in range(N) if known[i] > 0.5]
    if not cand:
        cand = [i for i in range(N) if alerts[i] > 0.5]
    return NOOP if not cand else 3 * random.choice(cand) + 1   # +1 = Restore

def oracle_policy(w):
    """Plafond : lit l'état VRAI (non observable par le DQN)."""
    nodes = sorted(w.ni.current_graph.get_nodes())
    comp = [i for i, n in enumerate(nodes) if n.true_compromised_status == 1]
    return NOOP if not comp else 3 * random.choice(comp) + 1
oracle_policy.oracle = True


POLICIES = {
    "random": random_policy,
    "noop": noop_policy,
    "restore_alerts": restore_alerts,
    "oracle": oracle_policy,
}


# ---------- évaluation ----------
def run_episode(policy, topo, seed, **gm_kw):
    random.seed(seed)
    np.random.seed(seed)
    env, ni, blue = make_env(topo, **gm_kw)
    w = DefenseEnv(env, ni, blue, seed=seed)
    obs = w.reset()
    ret, gt, steps, done = 0.0, 0.0, 0, False
    while not done:
        a = policy(w) if getattr(policy, "oracle", False) else policy(obs)
        out = w.step(a)
        if len(out) == 5:                      # (obs, r, terminated, truncated, info)
            obs, r, term, trunc, info = out
            done = term or trunc
        else:                                  # (obs, r, done, info)
            obs, r, done, info = out
        ret += r
        gt += info.get("gt", 0)
        steps += 1

    lost = steps < MAX_STEPS                   # défaite : tous les hôtes compromis
    gt_raw = gt
    if lost:                                   # les pas restants restent compromis
        gt -= (MAX_STEPS - steps)
    return ret, gt, gt_raw, steps, lost


def boot_ci(x, n_boot=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(out_csv="baselines_raw.csv", episodes=EPISODES, **gm_kw):
    rows = []
    for topo in TOPOS:
        for name, pol in POLICIES.items():
            for ep in range(episodes):
                ret, gt, gt_raw, steps, lost = run_episode(pol, topo, SEED0 + ep, **gm_kw)
                rows.append(dict(topo=topo, policy=name, seed=SEED0 + ep, ret=ret,
                                 gt=gt, gt_raw=gt_raw, steps=steps, lost=int(lost)))

    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["topo", "policy", "seed", "ret", "gt",
                                           "gt_raw", "steps", "lost"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'topo':8s} {'policy':15s} {'gt corrigé [IC95%]':>26s} "
          f"{'retour [IC95%]':>26s} {'% perdus':>9s}")
    for topo in TOPOS:
        for name in POLICIES:
            sel = [r for r in rows if r["topo"] == topo and r["policy"] == name]
            g = boot_ci([r["gt"] for r in sel])
            R = boot_ci([r["ret"] for r in sel])
            lost = 100 * np.mean([r["lost"] for r in sel])
            print(f"{topo:8s} {name:15s} {g[0]:7.1f} [{g[1]:6.1f},{g[2]:6.1f}]"
                  f" {R[0]:8.1f} [{R[1]:7.1f},{R[2]:7.1f}] {lost:8.0f}%")


if __name__ == "__main__":
    main()