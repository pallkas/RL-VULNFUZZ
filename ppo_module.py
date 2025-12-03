#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    PURE PYTORCH PPO MODULE FOR FUZZING                        ║
║                                                                               ║
║                    NO Gym, NO stable-baselines3                               ║
║                    Just PyTorch Neural Networks                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Author: Hunter (Shahid Lodin)
Version: 2.0.0

This module implements Proximal Policy Optimization (PPO) for guiding AFL++ 
fuzzing decisions. It learns to select optimal power schedules based on 
fuzzing state observations.

Architecture:
    Policy Network: 8 → 128 → 64 → 7 (actor)
    Value Network:  8 → 128 → 64 → 1 (critic)
    Total Parameters: ~22,000

State Space (8 dimensions):
    1. Total paths discovered (normalized)
    2. New paths rate
    3. Crashes found
    4. Hangs found
    5. Coverage percentage
    6. Execution speed
    7. Stability
    8. Current schedule

Action Space (7 discrete actions):
    0: explore  - Default exploration
    1: fast     - Fast coverage
    2: coe      - Cut-off exponential
    3: lin      - Linear
    4: quad     - Quadratic
    5: exploit  - Exploitation focused
    6: rare     - Rare branches

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    https://arxiv.org/abs/1707.06347
"""

import os
import random
import time
import subprocess
import signal
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# State space dimension (8 features from fuzzing state)
STATE_DIM = 8

# Action space dimension (7 AFL++ power schedules)
ACTION_DIM = 7

# Power schedule mapping
POWER_SCHEDULES = {
    0: "explore",
    1: "fast",
    2: "coe",
    3: "lin",
    4: "quad",
    5: "exploit",
    6: "rare"
}

# PPO Hyperparameters
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,              # Discount factor
    "epsilon_clip": 0.2,        # PPO clipping parameter
    "k_epochs": 4,              # Update epochs per batch
    "batch_size": 64,
    "hidden_dim": 128,
    "gae_lambda": 0.95,         # GAE parameter
    "entropy_coef": 0.01,       # Entropy bonus coefficient
    "value_coef": 0.5,          # Value loss coefficient
    "max_grad_norm": 0.5        # Gradient clipping
}

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    Actor network for PPO.
    
    Maps fuzzing state → action probabilities
    
    Architecture:
        Input (8) → Linear → ReLU → Linear → ReLU → Linear → Softmax
        8 → 128 → 64 → 7
    
    Total parameters: ~10,000
    """
    
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, 
                 hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for output layer
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Fuzzing state tensor [batch_size, 8]
        
        Returns:
            Action logits [batch_size, 7]
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax of logits)"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Critic network for PPO.
    
    Maps fuzzing state → state value estimate
    
    Architecture:
        Input (8) → Linear → ReLU → Linear → ReLU → Linear → Value
        8 → 128 → 64 → 1
    
    Total parameters: ~10,000
    """
    
    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Fuzzing state tensor [batch_size, 8]
        
        Returns:
            State value estimate [batch_size, 1]
        """
        return self.network(state)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class RolloutBuffer:
    """
    Buffer for storing rollout experiences.
    
    Stores complete trajectories for PPO updates.
    """
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            done: bool, log_prob: float, value: float):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns_and_advantages(self, last_value: float, 
                                        gamma: float = 0.99, 
                                        gae_lambda: float = 0.95):
        """
        Compute returns and GAE advantages.
        
        Uses Generalized Advantage Estimation (GAE) for reduced variance.
        """
        advantages = []
        returns = []
        gae = 0
        
        # Compute backwards from last step
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = last_value
                next_done = True
            else:
                next_value = self.values[step + 1]
                next_done = self.dones[step + 1]
            
            # TD error
            delta = (self.rewards[step] + 
                    gamma * next_value * (1 - next_done) - 
                    self.values[step])
            
            # GAE
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for training"""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch = {
                'states': torch.FloatTensor(np.array([self.states[i] for i in batch_indices])),
                'actions': torch.LongTensor([self.actions[i] for i in batch_indices]),
                'log_probs': torch.FloatTensor([self.log_probs[i] for i in batch_indices]),
                'advantages': torch.FloatTensor([self.advantages[i] for i in batch_indices]),
                'returns': torch.FloatTensor([self.returns[i] for i in batch_indices])
            }
            
            # Normalize advantages
            batch['advantages'] = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
            
            batches.append(batch)
        
        return batches
    
    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    
    Implements the PPO-Clip algorithm for learning fuzzing strategies.
    
    Key Components:
        - Policy Network (Actor): Selects actions
        - Value Network (Critic): Estimates state values
        - Clipped Surrogate Objective: Stable policy updates
        - GAE: Variance reduction for advantages
    
    Usage:
        agent = PPOAgent(state_dim=8, action_dim=7)
        
        # Get action
        action, log_prob, value = agent.get_action(state)
        
        # Store experience
        agent.buffer.add(state, action, reward, done, log_prob, value)
        
        # Update
        agent.update()
    """
    
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 config: Optional[Dict] = None):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary (uses defaults if None)
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        self.value = ValueNetwork(state_dim, self.config['hidden_dim']).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.config['learning_rate']
        )
        self.value_optimizer = optim.Adam(
            self.value.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            state: Current state tensor [1, state_dim] or [state_dim]
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            # Get action probabilities
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = self.value(state)
        
        return action.cpu(), log_prob.cpu(), value.cpu().squeeze()
    
    def get_action_deterministic(self, state: torch.Tensor) -> int:
        """
        Get deterministic action (for evaluation).
        
        Args:
            state: Current state tensor
        
        Returns:
            action: Action with highest probability
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state)
            action = torch.argmax(logits, dim=-1)
        
        return action.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Used during PPO update to get new log_probs and entropy.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
        
        Returns:
            log_probs: Log probabilities of actions under current policy
            values: State value estimates
            entropy: Policy entropy (for exploration bonus)
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Get action probabilities
        logits = self.policy(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Calculate log probs and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Get values
        values = self.value(states).squeeze(-1)
        
        return log_probs, values, entropy
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Uses clipped surrogate objective:
            L = min(r*A, clip(r, 1-ε, 1+ε)*A)
        
        where r = π_new / π_old (probability ratio)
        and A is the advantage estimate.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config['batch_size']:
            return {}
        
        # Get last value for GAE computation
        last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.value(last_state).item()
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value, 
            self.config['gamma'],
            self.config['gae_lambda']
        )
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.config['k_epochs']):
            for batch in self.buffer.get_batches(self.config['batch_size']):
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # Evaluate actions under current policy
                new_log_probs, values, entropy = self.evaluate_actions(states, actions)
                
                # Probability ratio
                ratios = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratios * advantages
                surr2 = torch.clamp(
                    ratios, 
                    1 - self.config['epsilon_clip'], 
                    1 + self.config['epsilon_clip']
                ) * advantages
                
                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config['value_coef'] * value_loss + 
                       self.config['entropy_coef'] * entropy_loss)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
                nn.utils.clip_grad_norm_(self.value.parameters(), self.config['max_grad_norm'])
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        self.training_step += 1
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'training_step': self.training_step
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
        print(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        policy_params = sum(p.numel() for p in self.policy.parameters())
        value_params = sum(p.numel() for p in self.value.parameters())
        
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'policy_parameters': policy_params,
            'value_parameters': value_params,
            'total_parameters': policy_params + value_params,
            'device': str(self.device),
            'training_steps': self.training_step,
            'config': self.config
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class FuzzingEnv:
    """
    Environment wrapper for AFL++ fuzzing.
    
    Provides RL interface to AFL++ fuzzer:
        - Observes fuzzing state (coverage, crashes, speed)
        - Takes actions (power schedule selection)
        - Computes rewards based on fuzzing progress
    
    Can run in two modes:
        - Real mode: Actually runs AFL++ (requires installation)
        - Simulation mode: Simulates fuzzing for training/testing
    """
    
    def __init__(self, target_binary: str = None, seed_dir: str = None,
                 output_dir: str = None, simulation: bool = True):
        """
        Initialize fuzzing environment.
        
        Args:
            target_binary: Path to target binary
            seed_dir: Directory containing initial seeds
            output_dir: Directory for fuzzing output
            simulation: If True, simulate fuzzing instead of running AFL++
        """
        self.target_binary = target_binary
        self.seed_dir = seed_dir
        self.output_dir = output_dir or f"./fuzz_output_{int(time.time())}"
        self.simulation = simulation or (target_binary is None)
        
        # State tracking
        self.current_schedule = 0
        self.step_count = 0
        self.total_paths = 0
        self.total_crashes = 0
        self.total_hangs = 0
        self.last_paths = 0
        self.exec_speed = 0
        self.coverage = 0.0
        self.stability = 100.0
        
        # AFL++ process
        self.afl_process: Optional[subprocess.Popen] = None
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_steps = 0
        self.max_steps = 1000
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        # Reset state
        self.step_count = 0
        self.total_paths = random.randint(0, 10) if self.simulation else 0
        self.total_crashes = 0
        self.total_hangs = 0
        self.last_paths = self.total_paths
        self.exec_speed = random.uniform(100, 500) if self.simulation else 0
        self.coverage = random.uniform(0, 10) if self.simulation else 0
        self.stability = 100.0
        self.current_schedule = 0
        self.episode_reward = 0
        self.episode_steps = 0
        
        # Stop any running AFL++ process
        if self.afl_process:
            self._stop_afl()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: Power schedule to use (0-6)
        
        Returns:
            next_state: New state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.step_count += 1
        self.episode_steps += 1
        
        # Apply action (change power schedule)
        self.current_schedule = action
        schedule_name = POWER_SCHEDULES.get(action, "explore")
        
        if self.simulation:
            self._simulate_step(action)
        else:
            self._real_step(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if done
        done = self.episode_steps >= self.max_steps
        
        # Get new state
        next_state = self._get_state()
        
        info = {
            'schedule': schedule_name,
            'total_paths': self.total_paths,
            'crashes': self.total_crashes,
            'exec_speed': self.exec_speed,
            'coverage': self.coverage
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns normalized 8-dimensional state vector.
        """
        # Calculate new paths rate
        new_paths = self.total_paths - self.last_paths
        self.last_paths = self.total_paths
        
        state = np.array([
            min(self.total_paths / 10000, 1.0),          # Total paths (normalized)
            min(new_paths / 100, 1.0),                    # New paths rate
            min(self.total_crashes / 100, 1.0),          # Crashes (normalized)
            min(self.total_hangs / 100, 1.0),            # Hangs (normalized)
            self.coverage / 100,                          # Coverage percentage
            min(self.exec_speed / 1000, 1.0),            # Execution speed (normalized)
            self.stability / 100,                         # Stability
            self.current_schedule / 6                     # Current schedule (normalized)
        ], dtype=np.float32)
        
        return state
    
    def _simulate_step(self, action: int):
        """Simulate fuzzing step (no actual AFL++)"""
        schedule = POWER_SCHEDULES.get(action, "explore")
        
        # Schedule effects on fuzzing
        schedule_effects = {
            "explore":  {"path_mult": 1.0, "speed_mult": 1.0, "crash_prob": 0.02},
            "fast":     {"path_mult": 0.8, "speed_mult": 1.3, "crash_prob": 0.015},
            "coe":      {"path_mult": 1.1, "speed_mult": 0.9, "crash_prob": 0.025},
            "lin":      {"path_mult": 1.0, "speed_mult": 1.0, "crash_prob": 0.02},
            "quad":     {"path_mult": 1.05, "speed_mult": 0.95, "crash_prob": 0.022},
            "exploit":  {"path_mult": 0.7, "speed_mult": 1.2, "crash_prob": 0.035},
            "rare":     {"path_mult": 1.3, "speed_mult": 0.7, "crash_prob": 0.03}
        }
        
        effects = schedule_effects.get(schedule, schedule_effects["explore"])
        
        # Simulate progress
        base_new_paths = random.randint(0, 5)
        self.total_paths += int(base_new_paths * effects["path_mult"])
        
        if random.random() < effects["crash_prob"]:
            self.total_crashes += 1
        
        if random.random() < 0.01:
            self.total_hangs += 1
        
        self.exec_speed = random.uniform(200, 500) * effects["speed_mult"]
        self.coverage = min(100, self.coverage + random.uniform(0, 0.5) * effects["path_mult"])
        self.stability = max(80, min(100, self.stability + random.uniform(-1, 1)))
    
    def _real_step(self, action: int):
        """Execute real AFL++ step"""
        # This would interact with actual AFL++ process
        # For now, just read stats from fuzzer_stats file
        stats_path = os.path.join(self.output_dir, "default", "fuzzer_stats")
        
        if os.path.exists(stats_path):
            stats = self._parse_fuzzer_stats(stats_path)
            self.total_paths = int(stats.get('corpus_count', self.total_paths))
            self.total_crashes = int(stats.get('saved_crashes', self.total_crashes))
            self.total_hangs = int(stats.get('saved_hangs', self.total_hangs))
            self.exec_speed = float(stats.get('execs_per_sec', self.exec_speed))
    
    def _parse_fuzzer_stats(self, path: str) -> Dict[str, str]:
        """Parse AFL++ fuzzer_stats file"""
        stats = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        stats[key.strip()] = value.strip()
        except Exception:
            pass
        return stats
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for current step.
        
        Reward components:
            - New paths discovered (primary goal)
            - Crashes found (valuable for vulnerability discovery)
            - Execution speed (efficiency)
            - Penalties for no progress
        """
        reward = 0.0
        
        # Reward for new paths (most important)
        new_paths = self.total_paths - self.last_paths
        reward += new_paths * 10.0
        
        # Reward for crashes (valuable discoveries)
        if self.total_crashes > 0:
            reward += self.total_crashes * 50.0
        
        # Small reward for maintaining good speed
        reward += self.exec_speed / 1000.0
        
        # Penalty for no progress
        if new_paths == 0 and self.step_count > 10:
            reward -= 1.0
        
        # Bonus for coverage milestones
        if self.coverage > 50:
            reward += 5.0
        if self.coverage > 80:
            reward += 10.0
        
        return reward
    
    def _start_afl(self, schedule: str):
        """Start AFL++ with specified schedule"""
        if self.afl_process:
            self._stop_afl()
        
        cmd = [
            "afl-fuzz",
            "-i", self.seed_dir,
            "-o", self.output_dir,
            "-p", schedule,
            "--", self.target_binary
        ]
        
        try:
            self.afl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        except Exception as e:
            print(f"Failed to start AFL++: {e}")
    
    def _stop_afl(self):
        """Stop AFL++ process"""
        if self.afl_process:
            try:
                os.killpg(os.getpgid(self.afl_process.pid), signal.SIGTERM)
                self.afl_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.afl_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            finally:
                self.afl_process = None
    
    def close(self):
        """Cleanup environment"""
        self._stop_afl()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo_fuzzer(target_binary: str = None, episodes: int = 100,
                     steps_per_episode: int = 200, save_path: str = "./models/ppo_fuzzer.pt",
                     simulation: bool = True, verbose: bool = True) -> PPOAgent:
    """
    Train PPO agent for fuzzing.
    
    Args:
        target_binary: Path to target binary (None for simulation)
        episodes: Number of training episodes
        steps_per_episode: Steps per episode
        save_path: Path to save trained model
        simulation: Use simulation mode
        verbose: Print training progress
    
    Returns:
        Trained PPOAgent
    """
    print("=" * 60)
    print("PPO FUZZER TRAINING")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Steps/Episode: {steps_per_episode}")
    print(f"Mode: {'Simulation' if simulation else 'Real AFL++'}")
    print(f"Save path: {save_path}")
    print("=" * 60)
    
    # Initialize
    env = FuzzingEnv(target_binary=target_binary, simulation=simulation)
    agent = PPOAgent()
    
    # Print model info
    info = agent.get_model_info()
    print(f"\nModel Parameters: {info['total_parameters']:,}")
    print(f"Device: {info['device']}")
    print()
    
    # Training loop
    best_reward = float('-inf')
    reward_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # Get action
            state_tensor = torch.FloatTensor(state)
            action, log_prob, value = agent.get_action(state_tensor)
            
            # Take step
            next_state, reward, done, info = env.step(action.item())
            episode_reward += reward
            
            # Store experience
            agent.buffer.add(
                state, 
                action.item(), 
                reward, 
                done, 
                log_prob.item(), 
                value.item()
            )
            
            state = next_state
            
            if done:
                break
        
        # PPO update
        if len(agent.buffer) >= agent.config['batch_size']:
            metrics = agent.update()
        else:
            metrics = {}
        
        # Track progress
        reward_history.append(episode_reward)
        agent.episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(save_path)
        
        # Progress output
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward:8.2f} | "
                  f"Best: {best_reward:8.2f} | "
                  f"Paths: {env.total_paths:5d} | "
                  f"Crashes: {env.total_crashes:3d}")
    
    # Final save
    agent.save(save_path)
    env.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final Best Reward: {best_reward:.2f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)
    
    return agent


def run_with_trained_ppo(model_path: str, target_binary: str = None,
                         duration_seconds: int = 3600, simulation: bool = True) -> Dict:
    """
    Run fuzzing with trained PPO model.
    
    Args:
        model_path: Path to trained model
        target_binary: Target binary to fuzz
        duration_seconds: How long to run
        simulation: Use simulation mode
    
    Returns:
        Dictionary of fuzzing results
    """
    print("=" * 60)
    print("PPO-GUIDED FUZZING")
    print("=" * 60)
    
    # Load agent
    agent = PPOAgent()
    agent.load(model_path)
    
    # Initialize environment
    env = FuzzingEnv(target_binary=target_binary, simulation=simulation)
    
    # Run fuzzing
    state = env.reset()
    start_time = time.time()
    total_reward = 0
    steps = 0
    
    schedule_counts = {s: 0 for s in POWER_SCHEDULES.values()}
    
    print("\nRunning PPO-guided fuzzing...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get action from trained model
            state_tensor = torch.FloatTensor(state)
            action = agent.get_action_deterministic(state_tensor)
            schedule = POWER_SCHEDULES.get(action, "explore")
            schedule_counts[schedule] += 1
            
            # Take step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Progress update
            if steps % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {steps:5d} | "
                      f"Time: {elapsed:6.1f}s | "
                      f"Paths: {env.total_paths:5d} | "
                      f"Crashes: {env.total_crashes:3d} | "
                      f"Schedule: {schedule}")
            
            if done:
                state = env.reset()
            else:
                state = next_state
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()
    
    # Results
    results = {
        'total_steps': steps,
        'total_reward': total_reward,
        'total_paths': env.total_paths,
        'total_crashes': env.total_crashes,
        'total_hangs': env.total_hangs,
        'duration': time.time() - start_time,
        'schedule_distribution': schedule_counts
    }
    
    print("\n" + "=" * 60)
    print("FUZZING RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN / CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for standalone PPO module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Fuzzer Module")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--run", type=str, help="Run with trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--target", type=str, help="Target binary")
    parser.add_argument("--save", type=str, default="./models/ppo_fuzzer.pt", help="Save path")
    parser.add_argument("--info", action="store_true", help="Show model info")
    
    args = parser.parse_args()
    
    if args.info:
        agent = PPOAgent()
        info = agent.get_model_info()
        print("\nPPO Model Information:")
        print("-" * 40)
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    elif args.train:
        train_ppo_fuzzer(
            target_binary=args.target,
            episodes=args.episodes,
            save_path=args.save,
            simulation=(args.target is None)
        )
    
    elif args.run:
        run_with_trained_ppo(
            model_path=args.run,
            target_binary=args.target,
            simulation=(args.target is None)
        )
    
    else:
        # Demo
        print("PPO Fuzzer Module - Demo Mode")
        print("-" * 40)
        print("Training for 50 episodes (simulation)...")
        print()
        
        agent = train_ppo_fuzzer(episodes=50, steps_per_episode=100)
        
        print("\nRunning inference for 30 seconds...")
        run_with_trained_ppo(
            model_path="./models/ppo_fuzzer.pt",
            duration_seconds=30
        )


if __name__ == "__main__":
    main()
