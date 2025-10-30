"""
Maze-based RL Environment using PyGame and Gymnasium.

This environment implements a maze where an agent must navigate to the exit
while managing health, hunger, and risk using the specified reward function.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from typing import Tuple, Optional, Dict, Any
from utils import (
    WORLD_WIDTH, WORLD_HEIGHT, GRAVITY, JUMP_FORCE, MOVE_SPEED, FRICTION,
    PLAYER_SIZE, MAX_HEALTH, MAX_HUNGER, MAX_RISK, NUM_ACTIONS, OBS_DIM,
    ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_REST, ACTION_EAT,
    reward_function, utility_function, pain_cost, normalize_observation,
    get_health_color, get_risk_color, calculate_distance, is_collision, clamp
)

# Import PyGame and visual effects only when needed
try:
    import pygame
    from visual_effects import ReplayBuffer, EnhancedHUD, VisualEffects
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("PyGame not available, running in headless mode")


class MazeRLEnv(gym.Env):
    """
    Maze-based environment with health, hunger, and risk mechanics.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, render_every: int = 10):
        super().__init__()
        
        self.render_mode = render_mode
        self.render_every = render_every
        self.render_count = 0
        
        # Initialize PyGame
        if self.render_mode == "human" and PYGAME_AVAILABLE:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
                pygame.display.set_caption("Maze RL Environment")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
                
                # Enhanced visual features
                self.replay_buffer = ReplayBuffer()
                self.enhanced_hud = EnhancedHUD(self.screen, self.font)
                self.visual_effects = VisualEffects()
            except pygame.error as e:
                print(f"PyGame display initialization failed: {e}")
                print("Running in headless mode...")
                self.render_mode = None
        elif self.render_mode == "human" and not PYGAME_AVAILABLE:
            print("PyGame not available, running in headless mode...")
            self.render_mode = None
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        
        # Game state
        self.player_pos = np.array([50, WORLD_HEIGHT - 50], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_health = MAX_HEALTH
        self.player_hunger = 0
        self.player_risk = 0
        self.on_ground = False
        
        # Maze elements
        self.walls = []
        self.food_items = []
        self.enemies = []
        self.exit_pos = None
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_count = 0
        self.step_count = 0
        self.maze_completed = False
        self.food_collected = 0
        self.total_food = 0
        
        # Exploration tracking
        self.position_history = []
        self.stuck_penalty = 0
        
        # Visual effects
        self.pain_flash = 0
        self.hunger_pulse = 0
        
        # Initialize maze
        self._generate_maze()
    
    def _generate_maze(self):
        """Generate a simple maze layout."""
        self.walls = []
        self.food_items = []
        self.enemies = []
        
        # Create maze walls (simple grid-based maze)
        wall_positions = [
            # Outer walls
            (0, 0, WORLD_WIDTH, 20),  # Top wall
            (0, WORLD_HEIGHT-20, WORLD_WIDTH, 20),  # Bottom wall
            (0, 0, 20, WORLD_HEIGHT),  # Left wall
            (WORLD_WIDTH-20, 0, 20, WORLD_HEIGHT),  # Right wall
            
            # Inner maze walls
            (200, 100, 20, 200),  # Vertical wall 1
            (400, 200, 20, 200),  # Vertical wall 2
            (600, 100, 20, 200),  # Vertical wall 3
            
            (100, 200, 200, 20),  # Horizontal wall 1
            (300, 300, 200, 20),   # Horizontal wall 2
            (500, 200, 200, 20),   # Horizontal wall 3
            (100, 400, 200, 20),   # Horizontal wall 4
            (500, 400, 200, 20),   # Horizontal wall 5
        ]
        
        for x, y, width, height in wall_positions:
            self.walls.append({
                'rect': pygame.Rect(x, y, width, height),
                'type': 'wall'
            })
        
        # Place food items strategically
        food_positions = [
            (150, 150),  # Top-left area
            (350, 250),  # Middle area
            (550, 150),  # Top-right area
            (150, 350),  # Bottom-left area
            (550, 350),  # Bottom-right area
        ]
        
        self.food_items = []
        for i, (x, y) in enumerate(food_positions):
            self.food_items.append({
                'pos': np.array([x, y], dtype=np.float32),
                'value': 20,
                'id': i
            })
        
        self.total_food = len(self.food_items)
        
        # Place enemies strategically (fewer enemies)
        enemy_positions = [
            (300, 250),  # Guards middle food
            (550, 350),  # Guards bottom-right food
        ]
        
        for i, (x, y) in enumerate(enemy_positions):
            self.enemies.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 15,
                'damage': 15,  # Meaningful damage - agent will take damage
                'patrol_range': 50,
                'direction': 1 if i % 2 == 0 else -1,
                'speed': 0.5
            })
        
        # Set exit position (top-right corner)
        self.exit_pos = np.array([WORLD_WIDTH - 50, 50], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset player state
        self.player_pos = np.array([50, WORLD_HEIGHT - 50], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_health = MAX_HEALTH
        self.player_hunger = 0
        self.player_risk = 0
        self.on_ground = False
        
        # Reset episode tracking
        self.episode_reward = 0
        self.step_count = 0
        self.maze_completed = False
        self.food_collected = 0
        self.position_history = []
        self.stuck_penalty = 0
        self.pain_flash = 0
        self.hunger_pulse = 0
        
        # Regenerate maze
        self._generate_maze()
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'health': self.player_health,
            'hunger': self.player_hunger,
            'risk': self.player_risk,
            'episode': self.episode_count,
            'maze_completed': self.maze_completed,
            'food_collected': self.food_collected,
            'total_food': self.total_food
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Store previous state for reward calculation
        prev_health = self.player_health
        prev_hunger = self.player_hunger
        prev_risk = self.player_risk
        
        # Execute action
        self._execute_action(action)
        
        # Update physics
        self._update_physics()
        
        # Track position for stuck detection
        self.position_history.append(tuple(self.player_pos))
        if len(self.position_history) > 50:
            self.position_history.pop(0)
        
        # Update internal states
        self._update_states()
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_health, prev_hunger, prev_risk)
        
        # Maze completion logic
        if not self.maze_completed:
            # Check if reached exit
            distance_to_exit = calculate_distance(self.player_pos, self.exit_pos)
            if distance_to_exit < 30:
                self.maze_completed = True
                
                # Base completion reward
                reward += 1000
                
                # Efficiency bonus based on health
                health_efficiency = self.player_health / MAX_HEALTH
                efficiency_reward = 500 * health_efficiency
                reward += efficiency_reward
                
                # Speed bonus (fewer steps = better)
                speed_bonus = max(0, 1000 - self.step_count) * 2
                reward += speed_bonus
                
                # Food collection bonus
                food_bonus = self.food_collected * 100
                reward += food_bonus
                
                print(f"🎉 MAZE COMPLETED!")
                print(f"   Food collected: {self.food_collected}/{self.total_food}")
                print(f"   Final health: {self.player_health:.1f}")
                print(f"   Steps taken: {self.step_count}")
                print(f"   Completion reward: {1000}")
                print(f"   Efficiency bonus: {efficiency_reward:.1f}")
                print(f"   Speed bonus: {speed_bonus}")
                print(f"   Food bonus: {food_bonus}")
                print(f"   Total bonus: {1000 + efficiency_reward + speed_bonus + food_bonus}")
        
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self.player_health <= 0 or self.maze_completed
        truncated = self.step_count >= 2000
        
        # Get observation
        obs = self._get_observation()
        
        # Store step in replay buffer
        if self.render_mode == "human":
            self.replay_buffer.add_step(
                self.player_pos.copy(), action, reward, 
                self.player_health, self.player_hunger, self.player_risk, self.player_pos.copy()
            )
            
            # End trajectory if episode is over
            if terminated or truncated:
                self.replay_buffer.end_trajectory()
        
        info = {
            'health': self.player_health,
            'hunger': self.player_hunger,
            'risk': self.player_risk,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'step': self.step_count,
            'maze_completed': self.maze_completed,
            'food_collected': self.food_collected,
            'total_food': self.total_food
        }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int):
        """Execute the given action."""
        if action == ACTION_LEFT:
            self.player_vel[0] -= MOVE_SPEED
        elif action == ACTION_RIGHT:
            self.player_vel[0] += MOVE_SPEED
        elif action == ACTION_JUMP and self.on_ground:
            self.player_vel[1] -= JUMP_FORCE
        elif action == ACTION_REST:
            # Rest reduces hunger slightly
            self.player_hunger = max(0, self.player_hunger - 1)
        elif action == ACTION_EAT:
            self._try_eat_food()
    
    def _update_physics(self):
        """Update physics simulation."""
        # Apply gravity
        self.player_vel[1] += GRAVITY
        
        # Apply friction
        self.player_vel[0] *= FRICTION
        
        # Update position
        self.player_pos += self.player_vel
        
        # Check wall collisions
        self._check_wall_collisions()
        
        # Keep player in bounds
        self.player_pos[0] = clamp(self.player_pos[0], PLAYER_SIZE//2, WORLD_WIDTH - PLAYER_SIZE//2)
        self.player_pos[1] = clamp(self.player_pos[1], PLAYER_SIZE//2, WORLD_HEIGHT - PLAYER_SIZE//2)
    
    def _check_wall_collisions(self):
        """Check collisions with walls."""
        player_rect = pygame.Rect(
            int(self.player_pos[0] - PLAYER_SIZE//2),
            int(self.player_pos[1] - PLAYER_SIZE//2),
            PLAYER_SIZE, PLAYER_SIZE
        )
        
        self.on_ground = False
        
        for wall in self.walls:
            if player_rect.colliderect(wall['rect']):
                # Handle collision
                if self.player_vel[1] > 0 and player_rect.bottom > wall['rect'].top:
                    # Landing on top
                    self.player_pos[1] = wall['rect'].top - PLAYER_SIZE//2
                    self.player_vel[1] = 0
                    self.on_ground = True
                elif self.player_vel[1] < 0 and player_rect.top < wall['rect'].bottom:
                    # Hitting from below
                    self.player_pos[1] = wall['rect'].bottom + PLAYER_SIZE//2
                    self.player_vel[1] = 0
                elif self.player_vel[0] > 0 and player_rect.right > wall['rect'].left:
                    # Hitting from left
                    self.player_pos[0] = wall['rect'].left - PLAYER_SIZE//2
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and player_rect.left < wall['rect'].right:
                    # Hitting from right
                    self.player_pos[0] = wall['rect'].right + PLAYER_SIZE//2
                    self.player_vel[0] = 0
    
    def _update_states(self):
        """Update health, hunger, and risk states."""
        # Increase hunger over time - meaningful rate for learning
        self.player_hunger = min(MAX_HUNGER, self.player_hunger + 0.3)
        
        # Check enemy collisions
        self._check_enemy_collisions()
        
        # Update risk based on enemy proximity
        self._update_risk()
    
    def _check_enemy_collisions(self):
        """Check collisions with enemies."""
        player_rect = pygame.Rect(
            int(self.player_pos[0] - PLAYER_SIZE//2),
            int(self.player_pos[1] - PLAYER_SIZE//2),
            PLAYER_SIZE, PLAYER_SIZE
        )
        
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(
                int(enemy['pos'][0] - enemy['size']//2),
                int(enemy['pos'][1] - enemy['size']//2),
                enemy['size'], enemy['size']
            )
            
            if player_rect.colliderect(enemy_rect):
                self.player_health -= enemy['damage']
                self.pain_flash = 10
                
                # Add pain visual effects
                if self.render_mode == "human":
                    for _ in range(5):
                        particle_pos = (self.player_pos[0] + random.randint(-10, 10),
                                      self.player_pos[1] + random.randint(-10, 10))
                        particle_vel = (random.uniform(-2, 2), random.uniform(-2, 2))
                        self.visual_effects.add_particle(particle_pos, (255, 0, 0), particle_vel, 20)
    
    def _update_risk(self):
        """Update risk based on enemy proximity."""
        min_distance = float('inf')
        for enemy in self.enemies:
            distance = calculate_distance(self.player_pos, enemy['pos'])
            min_distance = min(min_distance, distance)
        
        if min_distance < 100:
            risk_increase = (100 - min_distance) / 100 * 1
            self.player_risk = min(MAX_RISK, self.player_risk + risk_increase)
    
    def _try_eat_food(self):
        """Try to eat nearby food."""
        for i, food in enumerate(self.food_items):
            distance = calculate_distance(self.player_pos, food['pos'])
            if distance < 25:
                self.player_hunger = max(0, self.player_hunger - food['value'])
                self.player_health = min(MAX_HEALTH, self.player_health + food['value'] // 2)
                self.food_collected += 1
                
                # Add eating visual effects
                if self.render_mode == "human":
                    for _ in range(3):
                        particle_pos = (food['pos'][0] + random.randint(-5, 5),
                                      food['pos'][1] + random.randint(-5, 5))
                        particle_vel = (random.uniform(-1, 1), random.uniform(-1, 1))
                        self.visual_effects.add_particle(particle_pos, (0, 255, 0), particle_vel, 15)
                
                del self.food_items[i]
                break
    
    def _calculate_reward(self, action: int, prev_health: float, prev_hunger: float, prev_risk: float) -> float:
        """Calculate reward using the specified formula."""
        # Your original reward function: r_t = u(s_t, a_t; H_t) − c_pain(H_t, a_t) − λ·max(0, H* − H_{t+1}) − α·G_t − β·R_t
        
        # Utility function based on action and health
        utility = utility_function(action, self.player_health)
        
        # Pain cost based on action and health
        pain = pain_cost(action, self.player_health)
        
        # Health loss penalty: λ·max(0, H* − H_{t+1})
        health_loss = max(0, prev_health - self.player_health)
        health_penalty = 2.0 * health_loss  # λ = 2.0
        
        # Hunger penalty: α·G_t
        hunger_penalty = 0.1 * self.player_hunger  # α = 0.1
        
        # Risk penalty: β·R_t
        risk_penalty = 0.2 * self.player_risk  # β = 0.2
        
        # Total reward
        reward = utility - pain - health_penalty - hunger_penalty - risk_penalty
        
        # Add exploration bonus to prevent getting stuck
        exploration_bonus = self._calculate_exploration_bonus()
        reward += exploration_bonus
        
        return reward
    
    def _calculate_exploration_bonus(self) -> float:
        """Calculate exploration bonus to prevent getting stuck."""
        # Small bonus for moving
        movement_bonus = 0.1 if abs(self.player_vel[0]) > 0.5 else 0
        
        # Bonus for exploring different areas
        x_pos = self.player_pos[0]
        area_bonus = 0
        
        if x_pos < 200:  # Left area
            area_bonus = 0.05
        elif x_pos > 600:  # Right area
            area_bonus = 0.05
        elif 200 <= x_pos <= 600:  # Middle area
            area_bonus = 0.1
        
        # Penalty for getting stuck
        stuck_penalty = 0
        if len(self.position_history) >= 20:
            recent_positions = self.position_history[-20:]
            x_positions = [pos[0] for pos in recent_positions]
            y_positions = [pos[1] for pos in recent_positions]
            
            x_range = max(x_positions) - min(x_positions)
            y_range = max(y_positions) - min(y_positions)
            
            if x_range < 50 and y_range < 50:
                stuck_penalty = -0.5
        
        return movement_bonus + area_bonus + stuck_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.array([
            self.player_pos[0],      # x position
            self.player_pos[1],      # y position
            self.player_vel[0],      # x velocity
            self.player_vel[1],      # y velocity
            self.player_health,      # health
            self.player_hunger,      # hunger
            self.player_risk,        # risk
        ], dtype=np.float32)
        
        return normalize_observation(obs)
    
    def render(self):
        """Render the environment."""
        if self.render_mode != "human" or not PYGAME_AVAILABLE:
            return
        
        self.render_count += 1
        if self.render_count % self.render_every != 0:
            return
        
        # Clear screen
        self.screen.fill((20, 20, 60))  # Dark blue background
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, (100, 100, 100), wall['rect'])
        
        # Draw food items
        for food in self.food_items:
            pygame.draw.circle(self.screen, (255, 255, 0), 
                             (int(food['pos'][0]), int(food['pos'][1])), 8)
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(enemy['pos'][0]), int(enemy['pos'][1])), enemy['size']//2)
        
        # Draw exit
        if self.exit_pos is not None:
            pygame.draw.circle(self.screen, (0, 255, 0), 
                             (int(self.exit_pos[0]), int(self.exit_pos[1])), 15)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             (int(self.exit_pos[0]), int(self.exit_pos[1])), 15, 3)
        
        # Draw player
        player_color = get_health_color(self.player_health)
        pygame.draw.rect(self.screen, player_color, 
                        (int(self.player_pos[0] - PLAYER_SIZE//2),
                         int(self.player_pos[1] - PLAYER_SIZE//2),
                         PLAYER_SIZE, PLAYER_SIZE))
        
        # Draw HUD
        self.enhanced_hud.draw(
            self.player_health, self.player_hunger, self.player_risk,
            self.episode_reward, self.episode_count, self.step_count,
            tuple(self.player_pos), self.get_action_meanings()
        )
        
        # Draw maze completion info
        if self.maze_completed:
            completed_text = self.font.render("MAZE COMPLETED!", True, (0, 255, 0))
            self.screen.blit(completed_text, (10, 130))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def get_action_meanings(self) -> list:
        """Get action meanings for display."""
        return ["Left", "Right", "Jump", "Rest", "Eat"]
    
    def close(self):
        """Close the environment."""
        if self.render_mode == "human" and PYGAME_AVAILABLE:
            pygame.quit()
    
    def save_replay_data(self, filename: str):
        """Save replay data."""
        if self.render_mode == "human":
            self.replay_buffer.save(filename)
    
    def get_replay_buffer(self):
        """Get replay buffer for analysis."""
        if self.render_mode == "human":
            return self.replay_buffer
        return None
