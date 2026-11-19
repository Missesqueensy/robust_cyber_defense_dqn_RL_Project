# make_env.py
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv  

from my_network import build_network
from make_game_mode import make_game_mode


def make_env(name: str, **gm_kwargs):
    net = build_network(name)
    ni = NetworkInterface(make_game_mode(**gm_kwargs), net)
    red = RedInterface(ni)
    blue = BlueInterface(ni)
    env = GenericNetworkEnv(red, blue, ni, collect_additional_per_ts_data=False)
    return env, ni, blue

if __name__ == "__main__":
    import random
    env, ni, blue = make_env("chain")
    print("actions env :", blue.get_number_of_actions())         
    obs = env.reset()
    print("obs :", obs.shape)                                     
    for t in range(10):
        a = random.randrange(blue.get_number_of_actions())
        obs, r, done, _ = env.step(a)
        print(t, "action", a, "reward", round(float(r), 2), "done", done)