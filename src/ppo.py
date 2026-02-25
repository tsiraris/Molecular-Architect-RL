from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple
import numpy as np
import torch

@dataclass   # Dataclass Decorator to auto-generate __init__, __repr__, __eq__ etc.
class Transition:
    """ A single transition (step) in the environment. """
    state: object  # opaque (graph tensors etc.)
    action: int
    reward: float
    done: float
    log_prob: float
    value: float

class PPOBuffer:
    """
    Vectorized-environment PPO buffer with per-env GAE.
    We store transitions per-env, then compute GAE with proper bootstrapping using
    the last value estimate for each env after collection.
    """
    def __init__(self, device: torch.device, num_envs: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.device = device
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        """ Clear the buffer. """
        self._traj: List[List[Transition]] = [[] for _ in range(self.num_envs)]     # self._traj[e] is a List[Transition]
        self.states: List[object] = []                                              # states is a List[object]
        self.actions: torch.Tensor | None = None                                    # If self.actions is not None then it is a torch.Tensor
        self.rewards: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None
        self.log_probs: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def push(self, env_id: int, state, action: int, reward: float, done: float, log_prob: float, value: float) -> None:
        """ Add a transition to the buffer. """
        self._traj[int(env_id)].append(Transition(state, int(action), float(reward), float(done), float(log_prob), float(value)))

    def finalize(self, last_values: torch.Tensor) -> None:
        """
        Compute GAE per env and flatten trajectories into training tensors.
        last_values: shape [num_envs] value estimates for *current* obs after collection needed for bootstrapping GAE.
        """
        last_values = last_values.detach().float().to(self.device).view(-1)
        assert last_values.numel() == self.num_envs      # assert that last_values.shape == (self.num_envs,)

        all_states: List[object] = []
        all_actions, all_rewards, all_dones, all_logps, all_vals = [], [], [], [], []
        all_advs, all_rets = [], []

        # Compute GAE per env
        for e in range(self.num_envs):
            traj = self._traj[e]
            if len(traj) == 0:
                continue

            rewards = np.array([t.reward for t in traj], dtype=np.float32)
            dones = np.array([t.done for t in traj], dtype=np.float32)
            values = np.array([t.value for t in traj], dtype=np.float32)

            adv = np.zeros_like(rewards)
            gae = 0.0
            next_value = float(last_values[e].item())

            for i in reversed(range(len(rewards))):
                mask = 1.0 - dones[i]
                delta = rewards[i] + self.gamma * next_value * mask - values[i]     # TD error
                gae = delta + self.gamma * self.gae_lambda * mask * gae             # GAE
                adv[i] = gae
                next_value = values[i]

            ret = adv + values          # return

            for i, t in enumerate(traj):
                all_states.append(t.state)
                all_actions.append(t.action)
                all_rewards.append(t.reward)
                all_dones.append(t.done)
                all_logps.append(t.log_prob)
                all_vals.append(t.value)
                all_advs.append(float(adv[i]))
                all_rets.append(float(ret[i]))

        self.states = all_states
        self.actions = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        self.rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        self.dones = torch.tensor(all_dones, dtype=torch.float32, device=self.device)
        self.log_probs = torch.tensor(all_logps, dtype=torch.float32, device=self.device)
        self.values = torch.tensor(all_vals, dtype=torch.float32, device=self.device)
        self.advantages = torch.tensor(all_advs, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(all_rets, dtype=torch.float32, device=self.device)

    def get_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[torch.Tensor]:
        """ 
        Get batches of indices from the buffer. 
        Receives batch_size and shuffle as arguments and returns batches of indices.
        """
        assert self.actions is not None and self.advantages is not None and self.returns is not None and self.log_probs is not None
        n = self.actions.size(0)
        # shuffle indices if shuffle is True
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, batch_size):
            yield idx[start:start + batch_size] # returns a generator object that yields batches of indices
