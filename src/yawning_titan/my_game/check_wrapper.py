# check_wrapper.py
import random
import numpy as np
from make_env import make_env
from defense_env import DefenseEnv

for name in ("chain", "ring", "star", "bridge"):
    random.seed(0); np.random.seed(0)
    env, ni, blue = make_env(name)
    w = DefenseEnv(env, ni, blue, seed=0)
    obs = w.reset()
    assert obs.shape == (30,)
    ret, gt, t = 0.0, 0.0, 0
    while True:
        obs, r, done, info = w.step(random.randrange(w.n_actions))
        ret += r; gt += info["gt"]; t += 1
        if done:
            break
    print(f"{name:7s} pas={t} retour={ret:.1f} gt={gt:.0f} "
          f"terminal={info['terminal']} tronqué={info['truncated']}")