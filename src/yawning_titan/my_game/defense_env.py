# defense_env.py
import numpy as np
from reward import compute_reward

N = 10
N_ACTIONS = 3 * N + 1            # 31 : (Isolate, Restore, Patch) x 10 + No-Op
KINDS = ("isolate", "restore", "patch")


class DefenseEnv:
    n_actions = N_ACTIONS
    obs_dim = 3 * N              # compromission (10) + alertes (10) + isolation (10)

    def __init__(self, env, ni, blue, p_fa=0.05, seed=None):
        self.env, self.ni = env, ni
        self.p_fa = p_fa
        self.rng = np.random.default_rng(seed)
        # indices de l'env retrouvés par le nom des méthodes (pas codés en dur)
        names = {fn.__name__: i for i, fn in blue.action_dict.items()}
        self.t_idx = {"patch": names["reduce_node_vulnerability"],
                      "restore": names["restore_node"],
                      "isolate": names["isolate_node"]}
        glob = {fn.__name__: i for i, fn in blue.global_action_dict.items()}
        self.noop_idx = blue.deceptive_actions + glob["do_nothing"]
        self.base = blue.deceptive_actions + blue.number_global_action
        self.n_std = blue.number_of_actions

    def _obs(self, raw):
        raw = np.asarray(raw, dtype=np.float32).ravel()
        alerts = raw[N:2 * N].copy()
        false_alarms = (self.rng.random(N) < self.p_fa).astype(np.float32)
        alerts = np.maximum(alerts, false_alarms)          # fausses alertes, non persistantes
        iso = np.array([float(n.isolated) for n in self.ni.current_graph.get_nodes()],
                       dtype=np.float32)
        return np.concatenate([raw[:N], alerts, iso])

    def reset(self):
        return self._obs(self.env.reset())

    def step(self, a):
        assert 0 <= a < N_ACTIONS
        if a == N_ACTIONS - 1:
            kind, env_a, v = "noop", self.noop_idx, None
        else:
            v, k = divmod(a, 3)
            kind = KINDS[k]
            env_a = self.base + self.n_std * v + self.t_idx[kind]

        raw, _, done, _ = self.env.step(env_a)

        nodes = self.ni.current_graph.get_nodes()
        n_total = len(nodes)
        n_comp = sum(int(n.true_compromised_status) for n in nodes)
        terminal = (n_comp == n_total)         # défaite : les 10 hôtes compromis

        # Restore reconnecte l'hôte (YT ne le fait pas) ; sans effet si défaite
        if kind == "restore" and not terminal:
            self.ni.reconnect_node(sorted(self.ni.current_graph.get_nodes())[v])

        reward = compute_reward(n_comp, n_total, "noop" if terminal else kind)
        info = {"terminal": terminal,
                "truncated": bool(done) and not terminal,
                "n_comp": n_comp,
                "gt": -1.0 if n_comp > 0 else 0.0,     # score ground truth (Bates 2025)
                "kind": kind}
        return self._obs(raw), reward, bool(done), info