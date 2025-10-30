"""
Utility functions and constants for the 2D Platform RL Environment.
"""

import numpy as np
import torch
import math

# Reward function constants
ALPHA = 0.1  # Hunger penalty coefficient
BETA = 0.2   # Risk penalty coefficient  
LAMBDA = 2.0 # Health loss penalty coefficient
GAMMA_0 = 0.99  # Base discount factor

# Environment constants
WORLD_WIDTH = 800
WORLD_HEIGHT = 600
GRAVITY = 0.5
JUMP_FORCE = -12
MOVE_SPEED = 3
FRICTION = 0.8

# Player constants
PLAYER_SIZE = 20
MAX_HEALTH = 100
MAX_HUNGER = 100
MAX_RISK = 100

# Action space
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_JUMP = 2
ACTION_REST = 3
ACTION_EAT = 4
NUM_ACTIONS = 5

# Observation space dimensions: [x, y, vx, vy, H, G, R]
OBS_DIM = 7

def reward_function(utility, pain_cost, health_loss, hunger, risk, health_current, health_max):
    """
    Calculate reward using the specified formula:
    r_t = u(s_t, a_t; H_t) - c_pain(H_t, a_t) - λ·max(0, H* - H_{t+1}) - α·G_t - β·R_t
    
    Args:
        utility: Utility function value u(s_t, a_t; H_t)
        pain_cost: Pain cost c_pain(H_t, a_t)
        health_loss: Health loss (H* - H_{t+1})
        hunger: Current hunger level G_t
        risk: Current risk level R_t
        health_current: Current health H_t
        health_max: Maximum health H*
    
    Returns:
        Reward value
    """
    health_penalty = LAMBDA * max(0, health_max - health_current)
    hunger_penalty = ALPHA * hunger
    risk_penalty = BETA * risk
    
    reward = utility - pain_cost - health_penalty - hunger_penalty - risk_penalty
    return reward

def discount_factor(health, risk):
    """
    Calculate discount factor: γ_t = γ₀ * g_H(H_{t+1}) * g_R(R_{t+1})
    
    Args:
        health: Current health level
        risk: Current risk level
    
    Returns:
        Discount factor
    """
    # Health function: higher health = higher discount (longer horizon)
    g_h = health / MAX_HEALTH
    
    # Risk function: higher risk = lower discount (shorter horizon)
    g_r = 1.0 - (risk / MAX_RISK)
    
    return GAMMA_0 * g_h * g_r

def utility_function(action, health):
    """
    Calculate utility based on action and health.
    Higher health allows for more beneficial actions.
    
    Args:
        action: Action taken
        health: Current health level
    
    Returns:
        Utility value
    """
    base_utilities = {
        ACTION_LEFT: 0.1,
        ACTION_RIGHT: 0.1,
        ACTION_JUMP: 0.2,
        ACTION_REST: 0.5,
        ACTION_EAT: 0.3
    }
    
    health_factor = health / MAX_HEALTH
    return base_utilities[action] * health_factor

def pain_cost(action, health):
    """
    Calculate pain cost based on action and health.
    Actions are more painful when health is low.
    
    Args:
        action: Action taken
        health: Current health level
    
    Returns:
        Pain cost
    """
    base_costs = {
        ACTION_LEFT: 0.05,
        ACTION_RIGHT: 0.05,
        ACTION_JUMP: 0.1,
        ACTION_REST: 0.0,
        ACTION_EAT: 0.02
    }
    
    health_factor = 1.0 - (health / MAX_HEALTH)
    return base_costs[action] * (1.0 + health_factor)

def normalize_observation(obs):
    """
    Normalize observation to [-1, 1] range for better training.
    
    Args:
        obs: Raw observation array
    
    Returns:
        Normalized observation array
    """
    normalized = np.zeros_like(obs)
    
    # Position normalization (assuming world bounds)
    normalized[0] = (obs[0] - WORLD_WIDTH/2) / (WORLD_WIDTH/2)  # x
    normalized[1] = (obs[1] - WORLD_HEIGHT/2) / (WORLD_HEIGHT/2)  # y
    
    # Velocity normalization (assuming reasonable velocity bounds)
    max_vel = 15.0
    normalized[2] = np.clip(obs[2] / max_vel, -1, 1)  # vx
    normalized[3] = np.clip(obs[3] / max_vel, -1, 1)  # vy
    
    # Health, Hunger, Risk normalization (0-100 scale to -1,1)
    normalized[4] = (obs[4] / MAX_HEALTH) * 2 - 1  # health
    normalized[5] = (obs[5] / MAX_HUNGER) * 2 - 1  # hunger
    normalized[6] = (obs[6] / MAX_RISK) * 2 - 1    # risk
    
    return normalized

def get_health_color(health):
    """
    Get color based on health level for visual feedback.
    
    Args:
        health: Current health level (0-100)
    
    Returns:
        RGB color tuple
    """
    if health > 75:
        return (0, 255, 0)  # Green
    elif health > 50:
        return (255, 255, 0)  # Yellow
    elif health > 25:
        return (255, 165, 0)  # Orange
    else:
        return (255, 0, 0)  # Red

def get_risk_color(risk):
    """
    Get color based on risk level for visual feedback.
    
    Args:
        risk: Current risk level (0-100)
    
    Returns:
        RGB color tuple
    """
    intensity = int(255 * (risk / MAX_RISK))
    return (intensity, 0, 0)  # Red intensity

def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
    
    Returns:
        Distance value
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def is_collision(rect1, rect2):
    """
    Check if two rectangles are colliding.
    
    Args:
        rect1: First rectangle (x, y, width, height)
        rect2: Second rectangle (x, y, width, height)
    
    Returns:
        True if colliding, False otherwise
    """
    return (rect1[0] < rect2[0] + rect2[2] and
            rect1[0] + rect1[2] > rect2[0] and
            rect1[1] < rect2[1] + rect2[3] and
            rect1[1] + rect1[3] > rect2[1])

def clamp(value, min_val, max_val):
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))
