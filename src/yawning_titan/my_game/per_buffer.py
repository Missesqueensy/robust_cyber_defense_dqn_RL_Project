# per_buffer.py
import numpy as np
import torch


class SumTree:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, transition: tuple):
        idx = self.write + self.capacity - 1
        self.data[self.write] = transition
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s: float):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if s <= self.tree[left]:
                parent = left
            else:
                s -= self.tree[left]
                parent = right
        data_idx = leaf - self.capacity + 1
        return leaf, float(self.tree[leaf]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al., 2016)."""

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-5,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = 1.0
        self.frame = 1

    def __len__(self) -> int:
        return self.tree.size

    def _beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        # Nouvelle transition ajoutée avec priorité maximale pour garantir sa visite
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = []
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        total = self.tree.total()
        segment = total / batch_size
        beta = self._beta()
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf_idx, p, data = self.tree.get(s)
            # Sécurité si un arrondi flottant tombe sur une feuille vide
            while data is None:
                s = np.random.uniform(0.0, total)
                leaf_idx, p, data = self.tree.get(s)
            tree_indices[i] = leaf_idx
            priorities[i] = p
            batch.append(data)

        probs = priorities / total
        weights = (len(self) * probs) ** (-beta)
        weights /= weights.max()  # Normalisation pour stabilité du gradient

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
            tree_indices,
            torch.as_tensor(weights, dtype=torch.float32, device=device),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        td_abs = np.abs(td_errors) + self.eps
        for idx, p in zip(tree_indices, td_abs):
            self.max_priority = max(self.max_priority, float(p))
            self.tree.update(int(idx), float(p) ** self.alpha)