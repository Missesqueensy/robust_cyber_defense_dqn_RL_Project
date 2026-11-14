# diag2.py
import random
import numpy as np
from make_env import make_env
from defense_env import DefenseEnv

for p in (0.01, 1.0):
    env, ni, blue = make_env("chain", p_spread=p)
    print("valeur lue dans l'env :", p, ni.game_mode.red.natural_spreading.chance.to_connected_node.value)

    finals = []
    for e in range(10):
        random.seed(e); np.random.seed(e)
        env, ni, blue = make_env("chain", p_spread=p)
        w = DefenseEnv(env, ni, blue, seed=e)
        w.reset()
        for _ in range(20):
            _, _, done, info = w.step(30)      # 30 = No-Op
            if done:
                break
        finals.append(info["n_comp"])
    print("   hôtes compromis après 20 No-Op (moyenne sur 10 épisodes) :", np.mean(finals))