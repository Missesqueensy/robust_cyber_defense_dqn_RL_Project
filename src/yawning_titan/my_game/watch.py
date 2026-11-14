# watch.py
from make_env import make_env
from defense_env import DefenseEnv
from baselines import restore_alerts

env, ni, blue = make_env("star", p_spread=0.15)
w = DefenseEnv(env, ni, blue, seed=0)
obs, done = w.reset(), False
while not done:
    obs, r, done, info = w.step(restore_alerts(obs))
    env.render(show_node_names=True)