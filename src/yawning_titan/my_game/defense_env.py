# defense_env.py
from collections import deque
import numpy as np
from reward import compute_reward

N = 10
N_ACTIONS = 3 * N + 1            # 31 : (Isolate, Restore, Patch) x 10 + No-Op
KINDS = ("isolate", "restore", "patch")


class DefenseEnv:
    n_actions = N_ACTIONS

    def __init__(self, env, ni, blue, p_det=0.7, p_fa=0.05, history_len=3, seed=None):
        self.env, self.ni = env, ni
        self.p_det = p_det
        self.p_fa = p_fa
        self.history_len = history_len
        self.obs_dim = 3 * N * history_len   
        self.rng = np.random.default_rng(seed)

        names = {fn.__name__: i for i, fn in blue.action_dict.items()}
        self.t_idx = {
            "patch": names["reduce_node_vulnerability"],
            "restore": names["restore_node"],
            "isolate": names["isolate_node"],
        }
        glob = {fn.__name__: i for i, fn in blue.global_action_dict.items()}
        self.noop_idx = blue.deceptive_actions + glob["do_nothing"]
        self.base = blue.deceptive_actions + blue.number_global_action
        self.n_std = blue.number_of_actions

        self.prev_true_comp = np.zeros(N, dtype=np.float32)
        self.known_comp = np.zeros(N, dtype=np.float32)
        self.frames = deque(maxlen=history_len)

    def _single_obs(self, raw):
        raw = np.asarray(raw, dtype=np.float32).ravel()
        true_comp = raw[:N]

        # detection partielle:
        newly_comp = (true_comp == 1.0) & (self.prev_true_comp == 0.0)
        discovered = self.rng.random(N) < self.p_det
        self.known_comp = np.where(newly_comp, discovered.astype(np.float32), self.known_comp)
        # si un nœud n est plus compromis (restauré) on remet a 0
        self.known_comp = self.known_comp * true_comp
        self.prev_true_comp = true_comp.copy()

        # alertes IDPS + fausses alertes (p_fa)
        alerts = raw[N:2 * N].copy()
        false_alarms = (self.rng.random(N) < self.p_fa).astype(np.float32)
        alerts = np.maximum(alerts, false_alarms)

        # etat d'isoation des noeuds ordonné (0..9)
        ordered_nodes = sorted(self.ni.current_graph.get_nodes())
        iso = np.array([float(n.isolated) for n in ordered_nodes], dtype=np.float32)

        return np.concatenate([self.known_comp.copy(), alerts, iso])

    def reset(self):
        self.prev_true_comp = np.zeros(N, dtype=np.float32)
        self.known_comp = np.zeros(N, dtype=np.float32)
        obs0 = self._single_obs(self.env.reset())
        self.frames.clear()
        for _ in range(self.history_len):
            self.frames.append(obs0)
        return np.concatenate(list(self.frames))

    def step(self, a):
        assert 0 <= a < N_ACTIONS
        if a == N_ACTIONS - 1:
            kind, env_a, v = "noop", self.noop_idx, None
        else:
            v, k = divmod(a, 3)
            kind = KINDS[k]
            env_a = self.base + self.n_std * v + self.t_idx[kind]

        raw, _, done, _ = self.env.step(env_a)

        ordered_nodes = sorted(self.ni.current_graph.get_nodes())
        n_total = len(ordered_nodes)
        n_comp = sum(int(n.true_compromised_status) for n in ordered_nodes)
        terminal = (n_comp == n_total)         # Vriae fin -> les 10 hotes sont compromises
        truncated = bool(done) and not terminal  # troncature 

        # restautration et reconnexion
        if kind == "restore" and not terminal:
            self.ni.reconnect_node(ordered_nodes[v])
            self.known_comp[v] = 0.0

        reward = compute_reward(n_comp, n_total, kind)
        obs_t = self._single_obs(raw)
        self.frames.append(obs_t)

        info = {
            "terminal": terminal,
            "truncated": truncated,
            "n_comp": n_comp,
            "gt": -1.0 if n_comp > 0 else 0.0,
            "kind": kind,
        }
        return np.concatenate(list(self.frames)), reward, bool(done), info