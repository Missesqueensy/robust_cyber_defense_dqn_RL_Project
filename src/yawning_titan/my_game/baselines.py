# baselines.py
import random
import time
import numpy as np
from make_env import make_env
from defense_env import DefenseEnv, N

NOOP = 3 * N            # action 30


def random_policy(obs):
    return random.randrange(3 * N + 1)


def restore_alerts(obs):
    """Restaure un hôte vu compromis, sinon un hôte en alerte (sur la trame courante), sinon No-Op."""
    last = obs[-3 * N:]                     # Dernière observation (taille 30)
    comp = np.flatnonzero(last[:N] == 1)
    alerts = np.flatnonzero(last[N:2 * N] == 1)
    cand = comp if len(comp) else alerts
    if len(cand) == 0:
        return NOOP
    return 3 * int(random.choice(list(cand))) + 1      # Restore(v) = 3*v + 1


def evaluate(policy, topo, episodes=50, seed0=1000, **gm_kw):
    gts, rets, steps = [], [], 0
    t0 = time.time()
    for e in range(episodes):
        random.seed(seed0 + e)
        np.random.seed(seed0 + e)
        env, ni, blue = make_env(topo, **gm_kw)
        w = DefenseEnv(env, ni, blue, seed=seed0 + e)
        obs, gt, ret, done = w.reset(), 0.0, 0.0, False
        while not done:
            obs, r, done, info = w.step(policy(obs))
            gt += info["gt"]
            ret += r
            steps += 1
        gts.append(gt)
        rets.append(ret)
    dt = time.time() - t0
    return float(np.mean(gts)), float(np.std(gts)), float(np.mean(rets)), float(np.std(rets)), steps / dt


if __name__ == "__main__":
    for topo in ("chain", "ring", "star", "bridge"):
        for name, pol in (("aléatoire", random_policy), ("restaure-alertes", restore_alerts)):
            gt_m, gt_s, ret_m, ret_s, sps = evaluate(pol, topo, episodes=50)
            print(
                f"{topo:7s} {name:17s} | "
                f"gt={gt_m:6.1f}±{gt_s:4.1f} | "
                f"retour={ret_m:7.1f}±{ret_s:5.1f} | "
                f"pas/s={sps:6.0f}"
            )