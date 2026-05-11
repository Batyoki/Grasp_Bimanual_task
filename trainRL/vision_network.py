import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DualCameraPPO(nn.Module):
    def __init__(self, action_dim=9, state_dim=9):
        super().__init__()
        
        # Shared CNN Architecture for 84x84 RGB images
        def create_cnn():
            return nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
                nn.Flatten()
            )
            
        self.top_cnn = create_cnn()
        self.front_cnn = create_cnn()
        
        # CNN output (3136 features each) + Proprioceptive State
        fusion_dim = (3136 * 2) + state_dim
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(fusion_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def get_features(self, top, front, state):
        top = top.permute(0, 3, 1, 2).float() / 255.0
        front = front.permute(0, 3, 1, 2).float() / 255.0
        return torch.cat([self.top_cnn(top), self.front_cnn(front), state], dim=1)

    def get_action_and_value(self, top, front, state, action=None):
        features = self.get_features(top, front, state)
        mean = self.actor_mean(features)
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if action is None: 
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)
        
    def get_value(self, top, front, state):
        return self.critic(self.get_features(top, front, state))