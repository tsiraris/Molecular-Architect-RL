import math
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOBuffer:
    """
    Buffer for PPO transitions.
    Stores (state, action, reward, value, logp) and computes GAE + advantages.
    """

    def __init__(self, buffer_size: int, obs_shape: tuple, device: torch.device):
        self.device = device
        self.buf_size = buffer_size
        self.obs_shape = obs_shape

        # Storage
        self.states = torch.zeros((buffer_size, *obs_shape), device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.logp_old = torch.zeros(buffer_size, device=device)

        # GAE / advantage
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)

        self.ptr = 0
        self.full = False

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float, value: float, logp: float):
        """
        Add transition to buffer.
        """
        self.states[self.ptr].copy_(state)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logp_old[self.ptr] = logp

        self.ptr += 1
        if self.ptr >= self.buf_size:
            self.full = True

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        """
        Compute GAE advantage and returns.
        Must be called after buffer is full OR when episode ends.
        """
        adv = 0.0
        self.returns[-1] = last_value

        for step in reversed(range(self.ptr)):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] if step + 1 < self.ptr else last_value
                - self.values[step]
            )
            adv = delta + gamma * lam * adv
            self.advantages[step] = adv
            self.returns[step] = adv + self.values[step]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def clear(self):
        """
        Reset buffer for next update.
        """
        self.ptr = 0
        self.full = False
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.logp_old.zero_()
        self.advantages.zero_()
        self.returns.zero_()


def ppo_update(
    policy_net: nn.Module,
    value_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: PPOBuffer,
    clip_epsilon: float,
    target_kl: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    epochs: int,
):
    """
    Perform PPO update for one batch of transitions.
    """

    total_loss = 0.0
    kl_divs = []

    for _ in range(epochs):
        # Evaluate current policy on stored states
        dist = Categorical(logits=policy_net(buffer.states))
        logp = dist.log_prob(buffer.actions)
        ratio = torch.exp(logp - buffer.logp_old)

        # clipped surrogate objective
        adv = buffer.advantages
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # entropy bonus
        entropy = dist.entropy().mean()
        entropy_term = -entropy_coef * entropy

        # value loss
        values_pred = value_net(buffer.states).squeeze(-1)
        value_loss = value_coef * (buffer.returns - values_pred).pow(2).mean()

        # total loss
        loss = policy_loss + entropy_term + value_loss
        optimizer.zero_grad()
        loss.backward()

        # gradient norm clipping
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()

        # approximate KL divergence
        with torch.no_grad():
            kl = (buffer.logp_old - logp).mean().item()
            kl_divs.append(kl)

        # early stop if KL gets too large
        if kl > target_kl:
            break

    return total_loss / epochs, sum(kl_divs) / len(kl_divs)