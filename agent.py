"""
PPO Agent implementation for the 2D Platform RL Environment.

This module implements a Proximal Policy Optimization (PPO) agent using PyTorch
with MLP networks for both policy and value functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Any
from collections import deque
import random
from utils import OBS_DIM, NUM_ACTIONS, discount_factor


class MLPNetwork(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[1])
        )
        
        self.action_head = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and log standard deviation."""
        shared_features = self.shared_layers(obs)
        action_logits = self.action_head(shared_features)
        return action_logits, self.log_std.expand_as(action_logits)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        action_logits, log_std = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        # Calculate log probability for PPO
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob, log_probs


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[1]),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)


class PPOBuffer:
    """Buffer for storing PPO experience data."""
    
    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, action: int, reward: float, value: float, 
            log_prob: float, done: bool):
        """Add experience to buffer."""
        self.observations[self.ptr] = torch.from_numpy(obs).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all stored data."""
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'dones': self.dones[:self.size]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """PPO Agent implementation."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        hidden_dims: List[int] = [128, 128]
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Networks
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.value_net = ValueNetwork(obs_dim, hidden_dims).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Buffer
        self.buffer = PPOBuffer(2048, obs_dim, device)
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Get action from policy."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.policy_net.get_action(obs_tensor, deterministic)
            value = self.value_net(obs_tensor)
        
        return action, log_prob.item(), value.item()
    
    def store_experience(self, obs: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store experience in buffer."""
        self.buffer.add(obs, action, reward, value, log_prob, done)
    
    def compute_returns_and_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                                     dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE."""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute returns (discounted cumulative rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Compute advantages (returns - values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, epochs: int = 4) -> Dict[str, float]:
        """Update policy and value networks."""
        if self.buffer.size < 32:  # Need minimum batch size
            return {}
        
        data = self.buffer.get()
        obs = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        values = data['values']
        old_log_probs = data['log_probs']
        dones = data['dones']
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
        
        # Training epochs
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(epochs):
            # Get current policy outputs
            action_logits, log_std = self.policy_net(obs)
            new_values = self.value_net(obs)
            
            # Compute new log probabilities
            new_log_probs = F.log_softmax(action_logits, dim=-1)
            new_action_log_probs = new_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute entropy
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1).mean()
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
        
        # Update training stats
        self.training_stats['policy_loss'].append(np.mean(policy_losses))
        self.training_stats['value_loss'].append(np.mean(value_losses))
        self.training_stats['entropy_loss'].append(np.mean(entropy_losses))
        self.training_stats['total_loss'].append(
            np.mean(policy_losses) + self.value_coef * np.mean(value_losses) - self.entropy_coef * np.mean(entropy_losses)
        )
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(policy_losses) + self.value_coef * np.mean(value_losses) - self.entropy_coef * np.mean(entropy_losses)
        }
    
    def save(self, filepath: str, agent_name: str = None, environment: str = None, episode: int = None):
        """
        Save agent state with metadata.
        
        Args:
            filepath: Path to save the model
            agent_name: Optional agent name (e.g., "Bob", "Berta")
            environment: Optional environment identifier (e.g., "base", "sensing", "proximity_reward")
            episode: Optional episode number
        """
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        
        # Add metadata if provided
        if agent_name:
            save_dict['agent_name'] = agent_name
        if environment:
            save_dict['environment'] = environment
        if episode is not None:
            save_dict['episode'] = episode
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str, transfer_learning: bool = True):
        """
        Load agent state with support for transfer learning between environments.
        
        If obs_dim differs between checkpoint and current network, this implements
        transfer learning by:
        - Loading all compatible layers (hidden layers, output layer)
        - Keeping input layer randomly initialized to learn new observation features
        
        Args:
            filepath: Path to checkpoint
            transfer_learning: If True, allow loading from different obs_dim (transfer learning)
                              If False, raise error if obs_dim mismatch
        
        Returns:
            dict with metadata (agent_name, environment, episode) if available
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check for obs_dim mismatch
        checkpoint_policy = checkpoint['policy_net_state_dict']
        checkpoint_value = checkpoint['value_net_state_dict']
        
        # Get input layer keys (first linear layer in each network)
        policy_input_key = 'shared_layers.0.weight'
        value_input_key = 'network.0.weight'
        
        current_policy = self.policy_net.state_dict()
        current_value = self.value_net.state_dict()
        
        # Check if input dimensions match
        obs_dim_mismatch = False
        if policy_input_key in checkpoint_policy:
            checkpoint_obs_dim = checkpoint_policy[policy_input_key].shape[1]
            current_obs_dim = current_policy[policy_input_key].shape[1]
            
            if checkpoint_obs_dim != current_obs_dim:
                obs_dim_mismatch = True
                if not transfer_learning:
                    raise RuntimeError(
                        f"Observation dimension mismatch: checkpoint={checkpoint_obs_dim}, "
                        f"current={current_obs_dim}. Set transfer_learning=True for curriculum learning."
                    )
                
                # Transfer learning: load compatible layers, skip input layer
                print(f"🔄 Transfer Learning: Adapting from obs_dim={checkpoint_obs_dim} to obs_dim={current_obs_dim}")
                
                # Load policy network (skip input layer and its bias)
                policy_state = current_policy.copy()
                policy_input_bias_key = 'shared_layers.0.bias'
                for key in checkpoint_policy:
                    # Skip input layer weight and bias (they need new dimensions)
                    if key not in [policy_input_key, policy_input_bias_key] and key in policy_state:
                        if checkpoint_policy[key].shape == policy_state[key].shape:
                            policy_state[key] = checkpoint_policy[key]
                        else:
                            print(f"   ⚠️  Skipping {key} (shape mismatch: {checkpoint_policy[key].shape} vs {policy_state[key].shape})")
                self.policy_net.load_state_dict(policy_state)
                
                # Load value network (skip input layer and its bias)
                value_state = current_value.copy()
                value_input_bias_key = 'network.0.bias'
                for key in checkpoint_value:
                    # Skip input layer weight and bias (they need new dimensions)
                    if key not in [value_input_key, value_input_bias_key] and key in value_state:
                        if checkpoint_value[key].shape == value_state[key].shape:
                            value_state[key] = checkpoint_value[key]
                        else:
                            print(f"   ⚠️  Skipping {key} (shape mismatch: {checkpoint_value[key].shape} vs {value_state[key].shape})")
                self.value_net.load_state_dict(value_state)
                
                print(f"   ✅ Loaded compatible layers, input layer(s) randomly initialized for new observation space")
            else:
                # Full match, load everything
                self.policy_net.load_state_dict(checkpoint_policy)
                self.value_net.load_state_dict(checkpoint_value)
        else:
            # Fallback to original behavior if structure is different
            self.policy_net.load_state_dict(checkpoint_policy)
            self.value_net.load_state_dict(checkpoint_value)
        
        # Load optimizers (may need adjustment if transfer learning)
        if not obs_dim_mismatch:
            # Only load optimizers if full network match
            try:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            except Exception as e:
                print(f"   ⚠️  Could not load optimizer states: {e}")
                print(f"   ✅ Starting with fresh optimizers (will adapt quickly)")
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        # Return metadata
        metadata = {}
        if 'agent_name' in checkpoint:
            metadata['agent_name'] = checkpoint['agent_name']
        if 'environment' in checkpoint:
            metadata['environment'] = checkpoint['environment']
        if 'episode' in checkpoint:
            metadata['episode'] = checkpoint['episode']
        
        return metadata if metadata else None
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return self.training_stats.copy()


class AdaptivePPOAgent(PPOAgent):
    """PPO Agent with adaptive discount factor based on health and risk."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Get action with adaptive discount factor."""
        # Extract health and risk from observation (denormalized)
        health = (obs[4] + 1) * 50  # Convert from [-1,1] to [0,100]
        risk = (obs[6] + 1) * 50    # Convert from [-1,1] to [0,100]
        
        # Update gamma based on health and risk
        self.gamma = discount_factor(health, risk)
        
        return super().get_action(obs, deterministic)
    
    def compute_returns_and_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                                     dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns with adaptive discount factor."""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Use current gamma (which may have been updated by get_action)
        gamma = self.gamma
        
        # Compute returns (discounted cumulative rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Compute advantages (returns - values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
