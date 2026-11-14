# check_env.py
from yawning_titan.envs.generic.core.network_interface import NetworkInterface

from my_network import build_network
from make_game_mode import make_game_mode

TOPOLOGIES = ("chain", "ring", "star", "bridge")


def check(name: str) -> int:
    net = build_network(name)
    gm = make_game_mode()
    ni = NetworkInterface(gm, net)

    size = ni.get_observation_size()
    obs = ni.get_current_observation()
    print(f"{name:7s} taille déclarée = {size}, forme réelle = {obs.shape}")
    assert obs.shape == (size,), "taille déclarée != taille réelle"

    # Les vulnérabilités doivent changer d'un reset à l'autre
    ni.reset()
    v1 = list(ni.get_all_vulnerabilities().values())
    ni.reset()
    v2 = list(ni.get_all_vulnerabilities().values())
    print("   vulnérabilités tirées à chaque reset :", v1 != v2)
    print("   exemple :", [round(x, 2) for x in v2[:5]])

    # L'entrée doit rester le nœud 0
    entries = [n.name for n in ni.current_graph.entry_nodes]
    print("   entrée :", entries)
    return size


if __name__ == "__main__":
    sizes = {t: check(t) for t in TOPOLOGIES}
    print("\ntailles :", sizes)
    assert len(set(sizes.values())) == 1, "la taille d'observation dépend de la topologie"
    print("OK : même taille d'observation pour les 4 topologies")