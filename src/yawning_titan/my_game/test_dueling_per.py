# test_dueling_per.py
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from make_env import make_env
from defense_env import DefenseEnv
from dueling_dqn import DQN
from per_buffer import PrioritizedReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_update(online_net: torch.nn.Module, target_net: torch.nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def evaluate_greedy(online_net, topo: str, device: torch.device, episodes: int = 30, seed0: int = 1000):
    """Évaluation sans exploration (epsilon = 0) sur une topologie donnée."""
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
    return float(np.mean(gts)), float(np.mean(rets))


def train_dueling_per_test(
    episodes: int = 600,
    batch_size: int = 64,
    gamma: float = 0.95,
    lr: float = 1e-4,
    tau: float = 0.005,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 400,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env, ni, blue = make_env("chain")
    w = DefenseEnv(env, ni, blue, seed=seed)

    state_dim = w.obs_dim      # 90 (avec history_len=3)
    action_dim = w.n_actions   # 31

    online_net = DQN(state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True).to(device)
    target_net = DQN(state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(
        capacity=50_000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=episodes * 90,
    )

    rewards_hist = []
    gt_hist = []
    loss_hist = []
    td_hist = []

    for ep in range(episodes):
        state = w.reset()
        ep_reward = 0.0
        ep_gt = 0.0
        ep_losses = []
        ep_tds = []

        # Décroissance linéaire d'epsilon
        frac = min(1.0, ep / max(1, eps_decay_episodes))
        epsilon = eps_start + frac * (eps_end - eps_start)

        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online_net(s_t).argmax(dim=1).item())

            next_state, reward, done, info = w.step(action)

            # IMPORTANT : on passe info["terminal"] (vrai état terminal) et NON done (qui inclut truncated)
            buffer.push(state, action, reward, next_state, float(info["terminal"]))

            state = next_state
            ep_reward += reward
            ep_gt += info["gt"]

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, terminals, tree_indices, weights = buffer.sample(
                    batch_size, device
                )

                q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Target DQN standard (mettre use_double=True si souhaité)
                    next_q = target_net(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q * (1.0 - terminals)

                td_errors = targets - q_values
                elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
                loss = (weights * elementwise_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

                soft_update(online_net, target_net, tau=tau)

                td_np = td_errors.detach().cpu().numpy()
                buffer.update_priorities(tree_indices, td_np)

                ep_losses.append(float(loss.item()))
                ep_tds.append(float(np.mean(np.abs(td_np))))

        rewards_hist.append(ep_reward)
        gt_hist.append(ep_gt)
        loss_hist.append(np.mean(ep_losses) if ep_losses else 0.0)
        td_hist.append(np.mean(ep_tds) if ep_tds else 0.0)

        if (ep + 1) % 50 == 0:
            print(
                f"Ep {ep+1:3d}/{episodes} | eps={epsilon:.2f} | "
                f"Retour(moy50)={np.mean(rewards_hist[-50:]):7.2f} | "
                f"GT(moy50)={np.mean(gt_hist[-50:]):6.2f} | "
                f"|TD|={td_hist[-1]:.3f}"
            )

    print("\n--- Évaluation Zero-Shot (epsilon = 0) ---")
    for topo in ("chain", "ring", "star", "bridge"):
        gt_eval, ret_eval = evaluate_greedy(online_net, topo, device, episodes=30)
        print(f"{topo:7s} | GT moyen = {gt_eval:6.1f} | Retour moyen = {ret_eval:7.1f}")

    return rewards_hist, gt_hist, loss_hist, td_hist


def moving_average(x, w=25):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_results(rewards, gts, losses, tds, out_path="learning_curve_dueling_per.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    w = 25

    # 1. Récompense cumulée
    axes[0].plot(rewards, alpha=0.25, color="#1f77b4", label="Retour par épisode")
    ma_r = moving_average(rewards, w)
    axes[0].plot(range(w - 1, len(rewards)), ma_r, color="#1f77b4", linewidth=2, label=f"Moyenne mobile ({w} ép.)")
    axes[0].set_ylabel("Récompense cumulée")
    axes[0].set_title("Test d'apprentissage corrigé : Dueling DQN + PER (Chain, POMDP, FrameStack=3)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Score Ground Truth
    axes[1].plot(gts, alpha=0.25, color="#2ca02c", label="Score GT par épisode")
    ma_gt = moving_average(gts, w)
    axes[1].plot(range(w - 1, len(gts)), ma_gt, color="#2ca02c", linewidth=2, label=f"Moyenne mobile ({w} ép.)")
    axes[1].set_ylabel("Score Ground Truth (max 0)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Erreur TD et Loss
    axes[2].plot(tds, color="#d62728", alpha=0.85, label="Erreur TD moyenne |δ|")
    axes[2].plot(losses, color="#ff7f0e", alpha=0.85, label="Huber Loss pondérée (IS)")
    axes[2].set_xlabel("Épisodes")
    axes[2].set_ylabel("Erreur / Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nGraphe sauvegardé dans : {out_path}")
    plt.show()


if __name__ == "__main__":
    rewards, gts, losses, tds = train_dueling_per_test(episodes=600)
    plot_results(rewards, gts, losses, tds)