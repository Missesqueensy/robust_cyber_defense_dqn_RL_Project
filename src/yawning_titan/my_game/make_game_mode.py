# make_game_mode.py
import json
from pathlib import Path

from yawning_titan.game_modes.game_mode import GameMode

JSON_PATH = Path(__file__).parent / "game_modes.json"   # garde ton chemin actuel s'il marche
BASE_PRESET = "2"                                       # "Default Game Mode"

# --- Constantes à calibrer ---
MAX_STEPS = 100          # horizon fixe T
RED_SKILL = 0.5          # compétence de l'attaquant
P_SPREAD = 0.15          # proba de propagation par arête et par pas
P_ATTACK_ALERT = 0.5     # proba de détecter une attaque (alertes)


def make_game_mode(p_spread=P_SPREAD, skill=RED_SKILL) -> GameMode:
    with open(JSON_PATH) as f:
        d = json.load(f)["_default"][BASE_PRESET]
    d.pop("_doc_metadata", None)

    # Attaquant : attaque de base + propagation naturelle, pas d'attaque garantie
    red = d["red"]
    red["agent_attack"]["skill"] = {"use": True, "value": skill}
    for a in ("spread", "random_infect", "move", "do_nothing"):
        red["action_set"][a]["use"] = False
    red["action_set"]["basic_attack"] = {"use": True, "likelihood": 1}
    red["action_set"]["zero_day"]["use"] = False
    red["natural_spreading"] = {
        "capable": True,
        "chance": {"to_connected_node": p_spread, "to_unconnected_node": 0},
    }
    red["target_mechanism"]["random"] = True

    # Défenseur : Isolate, Restore, Patch par hôte + No-Op (reconnect exigé par YT, écarté par le wrapper)
    ba = d["blue"]["action_set"]
    ba["isolate_node"] = True
    ba["restore_node"] = True
    ba["reduce_vulnerability"] = True      # = Patch
    ba["reconnect_node"] = True
    ba["do_nothing"] = True
    ba["scan"] = False
    ba["make_node_safe"]["use"] = False
    ba["deceptive_nodes"] = {"use": False, "max_number": 0, "new_node_on_relocate": False}

    # Détection : sans scan, YT exige une vue de compromission parfaite
    d["blue"]["intrusion_discovery_chance"]["immediate"]["standard_node"] = 1
    ad = d["blue"]["attack_discovery"]
    ad["failed_attacks"]["chance"]["standard_node"] = P_ATTACK_ALERT
    ad["succeeded_attacks_known_compromise"]["chance"]["standard_node"] = P_ATTACK_ALERT
    ad["succeeded_attacks_unknown_compromise"]["chance"]["standard_node"] = P_ATTACK_ALERT

    # Règles : horizon fixe ; défaite seulement si les 10 hôtes sont compromis
    gr = d["game_rules"]
    gr["max_steps"] = MAX_STEPS
    gr["grace_period_length"] = 0
    gr["blue_loss_condition"] = {
        "all_nodes_lost": True,
        "high_value_node_lost": False,
        "target_node_lost": False,
        "n_percent_nodes_lost": {"use": False, "value": 0.8},
    }

    # Observation : compromission + alertes (l'isolation est ajoutée par le wrapper)
    d["observation_space"] = {
        "compromised_status": True,
        "attacked_nodes": True,
        "vulnerabilities": False,
        "attacking_nodes": False,
        "node_connections": False,
        "graph_connectivity": False,
        "average_vulnerability": False,
        "special_nodes": False,
        "red_agent_skill": False,
    }

    # Réinitialisation : entrée fixe, pas de HVN, vulnérabilités tirées à chaque épisode
    d["on_reset"] = {
        "randomise_vulnerabilities": True,
        "choose_new_high_value_nodes": False,
        "choose_new_entry_nodes": False,
    }

    # Récompense YT ignorée (le wrapper la recalcule)
    d["rewards"].update({
        "for_loss": 0, "for_reaching_max_steps": 0,
        "end_rewards_are_multiplied_by_end_state": False,
        "reduce_negative_rewards_for_closer_fails": False,
    })

    d["miscellaneous"]["random_seed"] = None
    return GameMode.create(d, raise_errors=True)


if __name__ == "__main__":
    gm = make_game_mode()
    print("node_connections:", gm.observation_space.node_connections.value)
    print("max_steps:", gm.game_rules.max_steps.value)