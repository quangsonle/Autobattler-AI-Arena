import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim=44):
        super().__init__()
        
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.move_head = nn.Linear(128, 5)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        feat = self.actor_backbone(state)
        # Clamping prevents saturated weights from ever overpowering exploration!
        move_logits = torch.clamp(self.move_head(feat), -10.0, 10.0)
        value = self.critic(state)
        return move_logits, value

    def act(self, state_tensor, deterministic=False, exploration_rate=0.0, temperature=1.0):
        # 1. Guaranteed True Random Exploration (Epsilon)
        if not deterministic and random.random() < exploration_rate:
            move_act = random.randint(0, 4)
            with torch.no_grad():
                val = self.critic(state_tensor).squeeze().item()
            return move_act, val

        # 2. Policy-driven action
        with torch.no_grad():
            move_logits, value = self.forward(state_tensor)
            if deterministic:
                move_act = torch.argmax(move_logits, dim=-1).item()
            else:
                t = max(0.05, float(temperature))
                scaled_logits = move_logits / t
                move_act = Categorical(logits=scaled_logits).sample().item()
            return move_act, value.squeeze().item()

    def evaluate_actions(self, states, move_acts):
        move_logits, values = self.forward(states)
        dist = Categorical(logits=move_logits)
        log_prob = dist.log_prob(move_acts)
        entropy = dist.entropy()
        return log_prob, values.squeeze(-1), entropy
