# run_ablation.py
import csv
import os
import random
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Évite tout crash graphique sans écran en fin de script
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from make_env import make_env
from defense_env import DefenseEnv
from dueling_dqn import DQN
from per_buffer import PrioritizedReplayBuffer


# etude factorielle 2^3
CONFIGS = [
    {"name": "DQN_Base",        "double": False, "dueling": False, "per": False},
    {"name": "Double",          "double": True,  "dueling": False, "per": False},
    {"name": "Dueling",         "double": False, "dueling": True,  "per": False},
    {"name": "PER",             "double": False, "dueling": False, "per": True},
    {"name": "Double_Dueling",  "double": True,  "dueling": True,  "per": False},
    {"name": "Double_PER",      "double": True,  "dueling": False, "per": True},
    {"name": "Dueling_PER",     "double": False, "dueling": True,  "per": True},
    {"name": "Full",            "double": True,  "dueling": True,  "per": True},
]

SEEDS = [0, 1, 2, 3, 4]
TOPOLOGIES = ("chain", "ring", "star", "bridge")

# hyperparamètres 
HP = {
    "episodes": 600,
    "batch_size": 64,
    "gamma": 0.95,
    "lr": 1e-4,
    "tau": 0.005,
    "hidden_dim": 256,
    "buffer_capacity": 50_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_episodes": 400,
    "eval_episodes": 50,
}

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
CURVES_CSV = RESULTS_DIR / "training_curves.csv"
EVAL_CSV = RESULTS_DIR / "zero_shot_eval.csv"


class UniformReplayBuffer:
    """Buffer de rejeu uniforme standard (lorsque per=False) avec la même interface."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action: int, reward: float, next_state, terminal: float):
        self.buffer.append((state, action, reward, next_state, terminal))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(terminals, dtype=torch.float32, device=device),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(online_net: torch.nn.Module, target_net: torch.nn.Module, tau: float):
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def evaluate_zero_shot(online_net, topo: str, device: torch.device, episodes: int = 50, seed0: int = 1000):
    """Évaluation Zero-Shot stricte (epsilon = 0) sur une topologie donnée."""
    online_net.eval()
    gts, rets = [], []
    for e in range(episodes):
        random.seed(seed0 + e)
        np.random.seed(seed0 + e)
        env, ni, blue = make_env(topo)
        w = DefenseEnv(env, ni, blue, seed=seed0 + e)
        state = w.reset()
        gt, ret, done = 0.0, 0.0, False
        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(online_net(s_t).argmax(dim=1).item())
            state, r, done, info = w.step(action)
            gt += info["gt"]
            ret += r
        gts.append(gt)
        rets.append(ret)
    online_net.train()
    return float(np.mean(gts)), float(np.std(gts)), float(np.mean(rets)), float(np.std(rets))


def train_single_run(cfg: dict, seed: int, device: torch.device):
    """Exécute 1 entraînement complet sur 'chain' puis l'évaluation Zero-Shot sur les 4 topologies."""
    set_seed(seed)
    env, ni, blue = make_env("chain")
    w = DefenseEnv(env, ni, blue, seed=seed)

    state_dim = w.obs_dim
    action_dim = w.n_actions

    online_net = DQN(state_dim, action_dim, hidden_dim=HP["hidden_dim"], enable_dueling_dqn=cfg["dueling"]).to(device)
    target_net = DQN(state_dim, action_dim, hidden_dim=HP["hidden_dim"], enable_dueling_dqn=cfg["dueling"]).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=HP["lr"])

    if cfg["per"]:
        buffer = PrioritizedReplayBuffer(
            capacity=HP["buffer_capacity"],
            alpha=0.6,
            beta_start=0.4,
            beta_frames=HP["episodes"] * 90,
        )
    else:
        buffer = UniformReplayBuffer(capacity=HP["buffer_capacity"])

    curve_rows = []
    t0 = time.time()

    for ep in range(HP["episodes"]):
        state = w.reset()
        ep_reward = 0.0
        ep_gt = 0.0
        ep_losses = []
        ep_tds = []

        frac = min(1.0, ep / max(1, HP["eps_decay_episodes"]))
        epsilon = HP["eps_start"] + frac * (HP["eps_end"] - HP["eps_start"])

        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online_net(s_t).argmax(dim=1).item())

            next_state, reward, done, info = w.step(action)
            buffer.push(state, action, reward, next_state, float(info["terminal"]))

            state = next_state
            ep_reward += reward
            ep_gt += info["gt"]

            if len(buffer) >= HP["batch_size"]:
                if cfg["per"]:
                    states, actions, rewards, next_states, terminals, tree_indices, weights = buffer.sample(
                        HP["batch_size"], device
                    )
                else:
                    states, actions, rewards, next_states, terminals = buffer.sample(HP["batch_size"], device)
                    weights = torch.ones(HP["batch_size"], device=device)

                q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if cfg["double"]:
                        next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
                        next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target_net(next_states).max(dim=1)[0]
                    targets = rewards + HP["gamma"] * next_q * (1.0 - terminals)

                td_errors = targets - q_values
                elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
                loss = (weights * elementwise_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

                soft_update(online_net, target_net, tau=HP["tau"])

                td_np = td_errors.detach().cpu().numpy()
                if cfg["per"]:
                    buffer.update_priorities(tree_indices, td_np)

                ep_losses.append(float(loss.item()))
                ep_tds.append(float(np.mean(np.abs(td_np))))

        curve_rows.append({
            "config": cfg["name"],
            "double": int(cfg["double"]),
            "dueling": int(cfg["dueling"]),
            "per": int(cfg["per"]),
            "seed": seed,
            "episode": ep + 1,
            "epsilon": round(epsilon, 4),
            "reward": round(ep_reward, 4),
            "gt": round(ep_gt, 4),
            "loss": round(float(np.mean(ep_losses)) if ep_losses else 0.0, 5),
            "td_error": round(float(np.mean(ep_tds)) if ep_tds else 0.0, 5),
        })

    # Sauvegarde du modèle entraîné
    model_path = MODELS_DIR / f"{cfg['name']}_seed{seed}.pt"
    torch.save(online_net.state_dict(), model_path)

    # Évaluation Zero-Shot (epsilon = 0) sur les 4 topologies
    eval_rows = []
    for topo in TOPOLOGIES:
        gt_m, gt_s, ret_m, ret_s = evaluate_zero_shot(
            online_net, topo, device, episodes=HP["eval_episodes"], seed0=1000
        )
        eval_rows.append({
            "config": cfg["name"],
            "double": int(cfg["double"]),
            "dueling": int(cfg["dueling"]),
            "per": int(cfg["per"]),
            "seed": seed,
            "topology": topo,
            "gt_mean": round(gt_m, 4),
            "gt_std": round(gt_s, 4),
            "reward_mean": round(ret_m, 4),
            "reward_std": round(ret_s, 4),
        })

    dt = time.time() - t0
    return curve_rows, eval_rows, dt


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Calcule la moyenne et l'intervalle de confiance bootstrap à 95%."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    low = float(np.percentile(means, 100 * (alpha / 2)))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(values.mean()), low, high


def generate_summary_and_plots():
    """Lit les CSV générés, affiche le tableau avec IC bootstrap 95% et trace les courbes moyennes."""
    import pandas as pd

    df_curves = pd.read_csv(CURVES_CSV)
    df_eval = pd.read_csv(EVAL_CSV)

    print("\n" + "=" * 85)
    print("SYNTHÈSE ZERO-SHOT (epsilon = 0) — MOYENNE ET IC BOOTSTRAP 95% SUR LES 5 GRAINES")
    print("=" * 85)
    summary_rows = []
    for cfg in CONFIGS:
        cname = cfg["name"]
        line = f"{cname:15s} |"
        for topo in TOPOLOGIES:
            sub = df_eval[(df_eval["config"] == cname) & (df_eval["topology"] == topo)]
            r_mean, r_lo, r_hi = bootstrap_ci(sub["reward_mean"].to_numpy())
            g_mean, g_lo, g_hi = bootstrap_ci(sub["gt_mean"].to_numpy())
            line += f" {topo[:3].upper()}: {r_mean:6.1f} [{r_lo:6.1f}, {r_hi:6.1f}] |"
            summary_rows.append({
                "config": cname,
                "topology": topo,
                "reward_mean": round(r_mean, 2),
                "reward_ci95_low": round(r_lo, 2),
                "reward_ci95_high": round(r_hi, 2),
                "gt_mean": round(g_mean, 2),
                "gt_ci95_low": round(g_lo, 2),
                "gt_ci95_high": round(g_hi, 2),
            })
        print(line)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary_bootstrap_ci95.csv", index=False)

    # Courbes d'apprentissage moyennes (moyenne mobile 25 épisodes + ±1 écart-type inter-graines)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    w = 25

    for cfg in CONFIGS:
        cname = cfg["name"]
        sub = df_curves[df_curves["config"] == cname]
        piv_r = sub.pivot(index="episode", columns="seed", values="reward").rolling(w, min_periods=1).mean()
        piv_g = sub.pivot(index="episode", columns="seed", values="gt").rolling(w, min_periods=1).mean()

        ep = piv_r.index.to_numpy()
        rm, rs = piv_r.mean(axis=1).to_numpy(), piv_r.std(axis=1).to_numpy()
        gm, gs = piv_g.mean(axis=1).to_numpy(), piv_g.std(axis=1).to_numpy()

        axes[0].plot(ep, rm, label=cname, linewidth=1.8)
        axes[0].fill_between(ep, rm - rs, rm + rs, alpha=0.12)

        axes[1].plot(ep, gm, label=cname, linewidth=1.8)
        axes[1].fill_between(ep, gm - gs, gm + gs, alpha=0.12)

    axes[0].axhline(-135.2, color="black", linestyle="--", linewidth=1.2, label="Heuristique restaure-alertes")
    axes[0].set_ylabel("Récompense cumulée (moy. mobile 25)")
    axes[0].set_title("Étude d'ablation 2³ sur Chain (5 graines, ±1 écart-type)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8.5)

    axes[1].axhline(-77.3, color="black", linestyle="--", linewidth=1.2, label="Heuristique restaure-alertes")
    axes[1].set_xlabel("Épisodes")
    axes[1].set_ylabel("Score Ground Truth (moy. mobile 25)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = RESULTS_DIR / "ablation_learning_curves.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\nTableaux CSV et graphique sauvegardés dans : {RESULTS_DIR.resolve()}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de la campagne d'ablation 2^3 (8 configs x {len(SEEDS)} seeds = {8 * len(SEEDS)} runs) sur {device}")

    # Détection des runs déjà complétés pour reprise automatique
    completed = set()
    if EVAL_CSV.exists():
        with open(EVAL_CSV, newline="") as f:
            for row in csv.DictReader(f):
                completed.add((row["config"], int(row["seed"])))

    curve_fields = ["config", "double", "dueling", "per", "seed", "episode", "epsilon", "reward", "gt", "loss", "td_error"]
    eval_fields = ["config", "double", "dueling", "per", "seed", "topology", "gt_mean", "gt_std", "reward_mean", "reward_std"]

    write_curve_header = not CURVES_CSV.exists()
    write_eval_header = not EVAL_CSV.exists()

    total_runs = len(CONFIGS) * len(SEEDS)
    run_idx = 0

    for cfg in CONFIGS:
        for seed in SEEDS:
            run_idx += 1
            if (cfg["name"], seed) in completed:
                print(f"[{run_idx:2d}/{total_runs}] {cfg['name']:15s} | seed={seed} -> déjà terminé, ignoré.")
                continue

            print(f"[{run_idx:2d}/{total_runs}] Entraînement {cfg['name']:15s} | seed={seed} ...", end=" ", flush=True)
            curve_rows, eval_rows, dt = train_single_run(cfg, seed, device)

            with open(CURVES_CSV, "a", newline="") as fc:
                writer = csv.DictWriter(fc, fieldnames=curve_fields)
                if write_curve_header:
                    writer.writeheader()
                    write_curve_header = False
                writer.writerows(curve_rows)

            with open(EVAL_CSV, "a", newline="") as fe:
                writer = csv.DictWriter(fe, fieldnames=eval_fields)
                if write_eval_header:
                    writer.writeheader()
                    write_eval_header = False
                writer.writerows(eval_rows)

            chain_res = next(r for r in eval_rows if r["topology"] == "chain")
            print(f"OK ({dt:.1f}s) | Chain Zero-Shot: Retour={chain_res['reward_mean']:.1f}, GT={chain_res['gt_mean']:.1f}")

    generate_summary_and_plots()


if __name__ == "__main__":
    main()