"""
2D Platform RL Environment with Food Proximity Reward Shaping.

This environment implements a 2D platformer where an agent must balance
health, hunger, and risk using the specified reward function.

DIFFERENCES:
1. REQUIRES ACTION_EAT - agent must learn to eat (no auto-eat)
2. Includes food proximity sensing in observations
3. Small proximity-based reward shaping (much smaller than food collection reward)

The proximity reward is designed so that eating food always outweighs
any negative proximity reward from moving away after eating.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from typing import Tuple, Optional, Dict, Any
from utils import (
    WORLD_WIDTH, WORLD_HEIGHT, GRAVITY, JUMP_FORCE, MOVE_SPEED, FRICTION,
    PLAYER_SIZE, MAX_HEALTH, MAX_HUNGER, MAX_RISK, NUM_ACTIONS,
    ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_REST, ACTION_EAT,
    reward_function, utility_function, pain_cost, normalize_observation,
    get_health_color, get_risk_color, calculate_distance, is_collision, clamp
)

# Extended observation dimension: base (7) + food sensing (3)
OBS_DIM_SENSING = 10

# Import PyGame and visual effects only when needed
try:
    import pygame
    from visual_effects import ReplayBuffer, EnhancedHUD, VisualEffects
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("PyGame not available, running in headless mode")


class Platform2DEnvProximityReward(gym.Env):
    """
    2D Platform environment with health, hunger, and risk mechanics.
    
    FEATURES:
    1. Requires ACTION_EAT to consume food (no auto-eat)
    2. Food proximity sensing in observations
    3. Proximity-based reward shaping (small, subservient to food collection)
    
    Observation space: [x, y, vx, vy, H, G, R, food_dist, food_dx, food_dy]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, render_every: int = 10, 
                 single_level: bool = True):
        super().__init__()
        
        self.render_mode = render_mode
        self.render_every = render_every
        self.render_count = 0
        self.single_level = single_level
        self.level_completion_reward = 2000  # Bonus reward for completing level
        self.efficiency_bonus = 1000  # Bonus for completing with high health
        
        # Initialize PyGame
        if self.render_mode == "human" and PYGAME_AVAILABLE:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
                pygame.display.set_caption("2D Platform RL Environment (Proximity Reward + Eat Action)")
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
        # Extended observation space with food proximity sensing
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM_SENSING,), dtype=np.float32
        )
        
        # Game state
        self.player_pos = np.array([WORLD_WIDTH // 2, WORLD_HEIGHT - 100], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_health = MAX_HEALTH
        self.player_hunger = 0
        self.player_risk = 0
        self.on_ground = False
        
        # Game objects
        self.platforms = []
        self.food_items = []
        self.enemies = []
        self.safe_zones = []
        
        # Episode tracking
        self.episode_reward = 0
        self.episode_count = 0
        self.step_count = 0
        self.level_completed = False
        self.food_collected = 0
        self.total_food = 0
        self.food_collected_this_step = False  # Track if food was collected this step
        self.prev_food_distance = None  # Track previous food distance for proximity reward
        
        # Exploration tracking
        self.position_history = []
        self.stuck_penalty = 0
        
        # Visual effects
        self.pain_flash = 0
        self.hunger_pulse = 0
        
        # Initialize world
        if self.single_level:
            self._generate_single_level()
        else:
            self._generate_world()
    
    def _generate_single_level(self):
        """Generate a single, well-designed level for mastery learning."""
        self.platforms = []
        self.food_items = []
        self.enemies = []
        self.safe_zones = []
        
        # Ground platform
        self.platforms.append({
            'rect': pygame.Rect(0, WORLD_HEIGHT - 20, WORLD_WIDTH, 20),
            'type': 'ground'
        })
        
        # Strategic platform layout - MORE platforms for better exploration
        platforms = [
            (100, 500, 120, 15),   # Left platform
            (250, 400, 120, 15),   # Middle-left platform  
            (450, 300, 120, 15),   # Middle platform
            (650, 200, 120, 15),   # Right platform
            (350, 150, 100, 15),   # Top platform (centered)
            (50, 350, 80, 15),     # Additional left platform
            (700, 100, 80, 15),    # Additional right platform
            (300, 250, 100, 15),   # Additional middle platform
        ]
        
        for x, y, width, height in platforms:
            self.platforms.append({
                'rect': pygame.Rect(x, y, width, height),
                'type': 'platform'
            })
        
        # FRIENDLY TRAINING ENVIRONMENT: More food, fewer enemies
        food_positions = [
            (160, 480),  # Left platform food
            (310, 380),  # Middle-left platform food
            (510, 280),  # Middle platform food
            (710, 180),  # Right platform food
            (400, 130),  # Top platform food (centered)
            (200, 450),  # Additional left food
            (600, 250),  # Additional right food
            (350, 350),  # Additional middle food
            # MORE FOOD FOR FRIENDLY TRAINING:
            (100, 400),  # Extra left food
            (250, 300),  # Extra left-middle food
            (450, 200),  # Extra middle food
            (650, 150),  # Extra right food
            (500, 400),  # Extra bottom-middle food
            (150, 200),  # Extra top-left food
            (750, 300),  # Extra top-right food
        ]
        
        
        self.food_items = []
        for i, (x, y) in enumerate(food_positions):
            self.food_items.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 12,
                'value': 25,  # Consistent food value
                'id': i  # Track which food item this is
            })
        
        self.total_food = len(self.food_items)
        
        # FRIENDLY ENEMY PLACEMENT: Fewer enemies, less threatening
        enemy_positions = [
            (510, 280),  # Only guards the most central food
            (710, 180),  # Only guards the rightmost food
            # Removed most enemies for friendlier training
        ]
        
        for i, (x, y) in enumerate(enemy_positions):
            self.enemies.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 12,  # Smaller enemies
                'damage': 10,  # Less damage
                'patrol_range': 20,  # Smaller patrol range
                'direction': 1 if i % 2 == 0 else -1,
                'speed': 0.2  # Even slower enemies
            })
        
        # MORE SAFE ZONES for friendlier training
        safe_zones = [
            (50, 300, 80, 50),   # Left safe zone
            (720, 100, 80, 50),  # Right safe zone
            (300, 200, 60, 40),  # Middle safe zone
            (600, 400, 60, 40),  # Bottom-right safe zone
        ]
        
        for x, y, width, height in safe_zones:
            self.safe_zones.append({
                'rect': pygame.Rect(x, y, width, height),
                'heal_rate': 0.05  # Slow healing - prevents safe zone exploit
            })
    
    def _generate_fixed_world(self):
        """Generate a fixed, learnable world layout."""
        self.platforms = []
        self.food_items = []
        self.enemies = []
        self.safe_zones = []
        
        # Ground platform
        self.platforms.append({
            'rect': pygame.Rect(0, WORLD_HEIGHT - 20, WORLD_WIDTH, 20),
            'type': 'ground'
        })
        
        # Fixed platform layout - designed to be learnable
        platform_layouts = {
            1: [  # Level 1: Simple platforming
                (100, 500, 120, 15),   # Left platform
                (300, 400, 120, 15),   # Middle platform  
                (500, 300, 120, 15),   # Right platform
                (700, 200, 120, 15),   # Top platform
            ],
            2: [  # Level 2: More complex
                (80, 450, 100, 15),   # Left platforms
                (200, 350, 100, 15),
                (400, 250, 100, 15),
                (600, 150, 100, 15),
                (750, 100, 100, 15),
            ],
            3: [  # Level 3: Advanced
                (50, 500, 80, 15),    # Complex layout
                (150, 400, 80, 15),
                (250, 300, 80, 15),
                (350, 200, 80, 15),
                (450, 100, 80, 15),
                (550, 150, 80, 15),
                (650, 250, 80, 15),
                (750, 350, 80, 15),
            ]
        }
        
        # Get layout for current level
        layout = platform_layouts.get(self.current_level, platform_layouts[1])
        
        for x, y, width, height in layout:
            self.platforms.append({
                'rect': pygame.Rect(x, y, width, height),
                'type': 'platform'
            })
        
        # Fixed food placement - strategic locations
        food_layouts = {
            1: [(150, 480), (350, 380), (550, 280), (750, 180)],  # On platforms
            2: [(130, 430), (230, 330), (430, 230), (630, 130), (780, 80)],
            3: [(90, 480), (190, 380), (290, 280), (390, 180), (490, 80), (590, 130), (690, 230), (790, 330)]
        }
        
        food_positions = food_layouts.get(self.current_level, food_layouts[1])
        for i, (x, y) in enumerate(food_positions):
            self.food_items.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 10,
                'value': 20 + (self.current_level * 5)  # More valuable food in higher levels
            })
        
        # Fixed enemy placement - strategic challenges
        enemy_layouts = {
            1: [(350, 380), (550, 280)],  # Few enemies, easy to avoid
            2: [(230, 330), (430, 230), (630, 130)],  # More enemies
            3: [(190, 380), (290, 280), (390, 180), (590, 130), (690, 230)]  # Many enemies
        }
        
        enemy_positions = enemy_layouts.get(self.current_level, enemy_layouts[1])
        for i, (x, y) in enumerate(enemy_positions):
            self.enemies.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 15,
                'damage': 10 + (self.current_level * 2),  # Stronger enemies in higher levels
                'patrol_range': 30,
                'direction': 1 if i % 2 == 0 else -1,
                'speed': 0.5 + (self.current_level * 0.2)
            })
        
        # Safe zones - fewer in higher levels
        safe_zone_layouts = {
            1: [(50, 300, 60, 40), (650, 100, 60, 40)],  # Two safe zones
            2: [(600, 50, 60, 40)],  # One safe zone
            3: [(700, 300, 60, 40)]   # One safe zone, harder to reach
        }
        
        safe_zones = safe_zone_layouts.get(self.current_level, safe_zone_layouts[1])
        for x, y, width, height in safe_zones:
            self.safe_zones.append({
                'rect': pygame.Rect(x, y, width, height),
                'heal_rate': 0.05  # Slow healing - prevents safe zone exploit
            })
    
    def _generate_world(self):
        """Generate random world layout."""
        self.platforms = []
        self.food_items = []
        self.enemies = []
        self.safe_zones = []
        
        # Ground platform
        self.platforms.append({
            'rect': pygame.Rect(0, WORLD_HEIGHT - 20, WORLD_WIDTH, 20),
            'type': 'ground'
        })
        
        # Random platforms
        for _ in range(8):
            x = random.randint(50, WORLD_WIDTH - 150)
            y = random.randint(200, WORLD_HEIGHT - 100)
            width = random.randint(80, 150)
            self.platforms.append({
                'rect': pygame.Rect(x, y, width, 15),
                'type': 'platform'
            })
        
        # Food items
        for _ in range(5):
            platform = random.choice(self.platforms[1:])  # Not ground
            x = platform['rect'].x + random.randint(10, platform['rect'].width - 20)
            y = platform['rect'].y - 15
            self.food_items.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 10,
                'value': random.randint(10, 30)
            })
        
        # Enemies
        for _ in range(3):
            platform = random.choice(self.platforms[1:])  # Not ground
            x = platform['rect'].x + random.randint(20, platform['rect'].width - 20)
            y = platform['rect'].y - 20
            self.enemies.append({
                'pos': np.array([x, y], dtype=np.float32),
                'size': 15,
                'damage': random.randint(5, 15),
                'patrol_range': 50,
                'direction': random.choice([-1, 1]),
                'speed': random.uniform(0.5, 1.5)
            })
        
        # Safe zones (healing areas)
        for _ in range(2):
            x = random.randint(50, WORLD_WIDTH - 100)
            y = random.randint(100, WORLD_HEIGHT - 200)
            self.safe_zones.append({
                'rect': pygame.Rect(x, y, 80, 60),
                'heal_rate': 0.05  # Slow healing - prevents safe zone exploit
            })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Update episode count if provided in options
        if options and 'episode' in options:
            self.episode_count = options['episode']
        
        # Reset player state
        self.player_pos = np.array([WORLD_WIDTH // 2, WORLD_HEIGHT - 100], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_health = MAX_HEALTH
        self.player_hunger = 0
        self.player_risk = 0
        self.on_ground = False
        
        # Reset episode tracking
        self.episode_reward = 0
        self.step_count = 0
        self.level_completed = False
        self.food_collected = 0
        self.position_history = []
        self.stuck_penalty = 0
        self.pain_flash = 0
        self.hunger_pulse = 0
        self.prev_food_distance = None  # Reset proximity tracking
        
        # Generate world (single level or random)
        if self.single_level:
            self._generate_single_level()
        else:
            self._generate_world()
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'health': self.player_health,
            'hunger': self.player_hunger,
            'risk': self.player_risk,
            'episode': self.episode_count,
            'level_completed': self.level_completed,
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
        
        # NO AUTO-EAT: Agent must use ACTION_EAT explicitly
        
        # Track position for stuck detection
        self.position_history.append(tuple(self.player_pos))
        if len(self.position_history) > 50:  # Keep last 50 positions
            self.position_history.pop(0)
        
        # Update internal states
        self._update_states()
        
        # Calculate base reward
        reward = self._calculate_reward(action, prev_health, prev_hunger, prev_risk)
        
        # PROXIMITY REWARD SHAPING - small reward for getting closer to food
        # This is MUCH smaller than food collection reward to prevent exploitation
        proximity_reward = self._calculate_proximity_reward()
        reward += proximity_reward
        
        # Debug print (can be removed later)
        if abs(proximity_reward) > 0.01:
            print(f"📏 Proximity reward: {proximity_reward:+.2f}")
        
        # DIRECT REWARD FOR FOOD COLLECTION - Big incentive! (check before resetting flag)
        if self.food_collected_this_step:
            # Base food reward
            food_reward = 100  # Significant direct reward
            
            # Bonus for eating when hungry (urgent need) - use PREVIOUS hunger before eating
            hunger_bonus = (prev_hunger / MAX_HUNGER) * 50  # Up to +50 if very hungry before eating
            
            # Bonus for eating when low health (critical need)
            health_bonus = ((MAX_HEALTH - prev_health) / MAX_HEALTH) * 50  # Up to +50 if low health before eating
            
            # Speed bonus - faster collection = better
            speed_bonus = max(0, 100 - (self.step_count / 10))  # Bonus for early collection
            
            total_food_reward = food_reward + hunger_bonus + health_bonus + speed_bonus
            reward += total_food_reward
            
            print(f"🍎 FOOD COLLECTED! Reward: +{total_food_reward:.1f} (base: {food_reward:.0f}, hunger: {hunger_bonus:.0f}, health: {health_bonus:.0f}, speed: {speed_bonus:.0f})")
            
            # Reset proximity tracking for next nearest food
            self.prev_food_distance = None
        
        # BOUNDARY PENALTY - continuous soft repelling force based on proximity
        boundary_penalty = self._calculate_boundary_penalty()
        reward += boundary_penalty
        
        # Reset food collection flag for next step
        self.food_collected_this_step = False
        
        # Level completion logic - complete when all food is collected (original version - no respawn)
        if self.single_level and not self.level_completed:
            # Check if all food has been collected
            if self.food_collected >= self.total_food:
                self.level_completed = True
                
                # Base completion reward
                reward += self.level_completion_reward
                
                # Efficiency bonus based on health
                health_efficiency = self.player_health / MAX_HEALTH
                efficiency_reward = self.efficiency_bonus * health_efficiency
                reward += efficiency_reward
                
                # Speed bonus (fewer steps = better)
                speed_bonus = max(0, 500 - self.step_count) * 2
                reward += speed_bonus
                
                print(f"🎉 LEVEL COMPLETED!")
                print(f"   Food collected: {self.food_collected}/{self.total_food}")
                print(f"   Final health: {self.player_health:.1f}")
                print(f"   Steps taken: {self.step_count}")
                print(f"   Completion reward: {self.level_completion_reward}")
                print(f"   Efficiency bonus: {efficiency_reward:.1f}")
                print(f"   Speed bonus: {speed_bonus}")
                print(f"   Total bonus: {self.level_completion_reward + efficiency_reward + speed_bonus}")
        
        self.episode_reward += reward
        
        # ORIGINAL: Episode ends when health reaches 0
        terminated = self.player_health <= 0  # Episode ends when health reaches 0
        truncated = False  # No artificial step limit - agent lives as long as it has health
        
        # Get observation
        obs = self._get_observation()
        
        # Store step in replay buffer
        if self.render_mode == "human":
            self.replay_buffer.add_step(
                obs, action, reward, 
                self.player_health, self.player_hunger, self.player_risk,
                tuple(self.player_pos)
            )
            
            # Add to HUD trajectory
            self.enhanced_hud.add_position(tuple(self.player_pos))
            self.enhanced_hud.add_action(action, self.get_action_meanings())
            
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
            'level_completed': self.level_completed,
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
            self.player_vel[1] = JUMP_FORCE
            self.on_ground = False
        elif action == ACTION_REST:
            # Resting increases hunger slightly (encourages exploration)
            self.player_hunger = min(MAX_HUNGER, self.player_hunger + 0.5)
            self.player_risk = max(0, self.player_risk - 0.5)
        elif action == ACTION_EAT:
            # Agent must explicitly use ACTION_EAT to consume food
            self._try_eat_food()
    
    def _update_physics(self):
        """Update physics simulation."""
        # Apply gravity
        if not self.on_ground:
            self.player_vel[1] += GRAVITY
        
        # Apply friction
        self.player_vel[0] *= FRICTION
        
        # Update position
        self.player_pos += self.player_vel
        
        # Check platform collisions
        self._check_platform_collisions()
        
        # Check world boundaries (hard collision - still prevents going outside)
        min_x = PLAYER_SIZE//2
        max_x = WORLD_WIDTH - PLAYER_SIZE//2
        
        if self.player_pos[0] < min_x:
            self.player_pos[0] = min_x
            if self.player_vel[0] < 0:
                self.player_vel[0] = 0
        
        if self.player_pos[0] > max_x:
            self.player_pos[0] = max_x
            if self.player_vel[0] > 0:
                self.player_vel[0] = 0
        
        if self.player_pos[1] > WORLD_HEIGHT:
            self.player_health = 0  # Fall damage
    
    def _check_platform_collisions(self):
        """Check collisions with platforms."""
        self.on_ground = False
        player_rect = pygame.Rect(
            int(self.player_pos[0] - PLAYER_SIZE//2),
            int(self.player_pos[1] - PLAYER_SIZE//2),
            PLAYER_SIZE, PLAYER_SIZE
        )
        
        for platform in self.platforms:
            if is_collision(
                (player_rect.x, player_rect.y, player_rect.width, player_rect.height),
                (platform['rect'].x, platform['rect'].y, platform['rect'].width, platform['rect'].height)
            ):
                # Landing on top of platform
                if (self.player_vel[1] > 0 and 
                    player_rect.bottom > platform['rect'].top and
                    player_rect.top < platform['rect'].top):
                    self.player_pos[1] = platform['rect'].top - PLAYER_SIZE//2
                    self.player_vel[1] = 0
                    self.on_ground = True
                # Hit platform from side
                elif self.player_vel[0] > 0 and player_rect.right > platform['rect'].left:
                    self.player_pos[0] = platform['rect'].left - PLAYER_SIZE//2
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and player_rect.left < platform['rect'].right:
                    self.player_pos[0] = platform['rect'].right + PLAYER_SIZE//2
                    self.player_vel[0] = 0
    
    def _update_states(self):
        """Update health, hunger, and risk states."""
        # Increase hunger over time - moderate rate for meaningful learning
        self.player_hunger = min(MAX_HUNGER, self.player_hunger + 0.3)
        
        # STARVATION MECHANICS: Continuous negative health for infinite learning
        if self.player_hunger >= MAX_HUNGER:
            # Very slow starvation damage - allows extended high-risk behavior observation
            starvation_damage = 0.1  # Reduced from 0.5 to 0.1 (5x slower)
            self.player_health = max(0, self.player_health - starvation_damage)
        
        # Check enemy collisions
        self._check_enemy_collisions()
        
        # Check safe zone effects
        self._check_safe_zones()
        
        # Update risk based on proximity to enemies
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
            
            if is_collision(
                (player_rect.x, player_rect.y, player_rect.width, player_rect.height),
                (enemy_rect.x, enemy_rect.y, enemy_rect.width, enemy_rect.height)
            ):
                self.player_health = max(0, self.player_health - enemy['damage'])
                self.pain_flash = 10  # Flash red for pain
                self.player_risk = min(MAX_RISK, self.player_risk + enemy['damage'])
                
                # Add visual effects
                if self.render_mode == "human":
                    self.visual_effects.add_screen_shake(15)
                    self.visual_effects.add_flash((255, 0, 0), 8)
                    # Add damage particles
                    for _ in range(5):
                        particle_pos = (self.player_pos[0] + random.randint(-10, 10),
                                      self.player_pos[1] + random.randint(-10, 10))
                        particle_vel = (random.uniform(-2, 2), random.uniform(-2, 2))
                        self.visual_effects.add_particle(particle_pos, (255, 0, 0), particle_vel, 20)
    
    def _check_safe_zones(self):
        """Check if player is in safe zones."""
        player_rect = pygame.Rect(
            int(self.player_pos[0] - PLAYER_SIZE//2),
            int(self.player_pos[1] - PLAYER_SIZE//2),
            PLAYER_SIZE, PLAYER_SIZE
        )
        
        for safe_zone in self.safe_zones:
            if is_collision(
                (player_rect.x, player_rect.y, player_rect.width, player_rect.height),
                (safe_zone['rect'].x, safe_zone['rect'].y, safe_zone['rect'].width, safe_zone['rect'].height)
            ):
                self.player_health = min(MAX_HEALTH, self.player_health + safe_zone['heal_rate'])
                self.player_risk = max(0, self.player_risk - 1)
    
    def _update_risk(self):
        """Update risk based on proximity to enemies and health status."""
        # Base risk from enemy proximity
        min_distance = float('inf')
        for enemy in self.enemies:
            distance = calculate_distance(self.player_pos, enemy['pos'])
            min_distance = min(min_distance, distance)
        
        # Risk increases as distance to nearest enemy decreases
        if min_distance < 100:
            risk_increase = (100 - min_distance) / 100 * 2
            self.player_risk = min(MAX_RISK, self.player_risk + risk_increase)
        
        # DRAMATIC HEALTH-BASED RISK: Much more aggressive desperation
        if self.player_health < 80:  # When health is below 80% (was 75%)
            desperation_factor = (80 - self.player_health) / 80  # 0 to 1
            desperation_risk = desperation_factor * 50  # Up to 50 risk points (was 25)
            self.player_risk = min(MAX_RISK, self.player_risk + desperation_risk)
        
        # DRAMATIC STARVATION RISK: Much more aggressive hunger-based risk
        if self.player_hunger > 50:  # When moderately hungry (was 60)
            starvation_factor = (self.player_hunger - 50) / 50  # 0 to 1 (was /40)
            starvation_risk = starvation_factor * 40  # Up to 40 risk points (was 20)
            self.player_risk = min(MAX_RISK, self.player_risk + starvation_risk)
    
    def _try_eat_food(self):
        """Try to eat nearby food."""
        for i, food in enumerate(self.food_items):
            distance = calculate_distance(self.player_pos, food['pos'])
            if distance < 25:  # Within eating range
                # SIGNIFICANT HEALTH/HUNGER RESTORATION - like respawning!
                prev_hunger = self.player_hunger
                prev_health = self.player_health
                
                # Food restores much more (equivalent to respawning health)
                hunger_restored = food['value'] * 2  # Double restoration
                health_restored = food['value']  # Full value, not half
                
                self.player_hunger = max(0, self.player_hunger - hunger_restored)
                self.player_health = min(MAX_HEALTH, self.player_health + health_restored)
                
                self.food_collected += 1  # Track food collection
                self.food_collected_this_step = True  # Mark for reward calculation
                
                # Add eating visual effects
                if self.render_mode == "human":
                    # Add healing particles
                    for _ in range(3):
                        particle_pos = (food['pos'][0] + random.randint(-5, 5),
                                      food['pos'][1] + random.randint(-5, 5))
                        particle_vel = (random.uniform(-1, 1), random.uniform(-1, 1))
                        self.visual_effects.add_particle(particle_pos, (0, 255, 0), particle_vel, 15)
                
                del self.food_items[i]
                break
    
    def _calculate_reward(self, action: int, prev_health: float, prev_hunger: float, prev_risk: float) -> float:
        """Calculate reward using the specified formula."""
        utility = utility_function(action, self.player_health)
        pain = pain_cost(action, self.player_health)
        health_loss = max(0, prev_health - self.player_health)
        
        reward = reward_function(
            utility, pain, health_loss, 
            self.player_hunger, self.player_risk,
            self.player_health, MAX_HEALTH
        )
        
        # Add exploration bonus to prevent getting stuck
        exploration_bonus = self._calculate_exploration_bonus()
        reward += exploration_bonus
        
        return reward
    
    def _calculate_exploration_bonus(self) -> float:
        """Calculate exploration bonus to prevent getting stuck."""
        # Small bonus for moving (encourages exploration)
        movement_bonus = 0.1 if abs(self.player_vel[0]) > 0.5 else 0
        
        # Bonus for being in different areas of the map
        x_pos = self.player_pos[0]
        area_bonus = 0
        
        if x_pos < 200:  # Left area
            area_bonus = 0.05
        elif x_pos > 600:  # Right area
            area_bonus = 0.05
        elif 200 <= x_pos <= 600:  # Middle area
            area_bonus = 0.1
        
        # Penalty for getting stuck (staying in same small area)
        stuck_penalty = 0
        if len(self.position_history) >= 20:
            recent_positions = self.position_history[-20:]
            x_positions = [pos[0] for pos in recent_positions]
            y_positions = [pos[1] for pos in recent_positions]
            
            x_range = max(x_positions) - min(x_positions)
            y_range = max(y_positions) - min(y_positions)
            
            # If agent hasn't moved much in last 20 steps, apply penalty
            if x_range < 50 and y_range < 50:
                stuck_penalty = -0.5
        
        return movement_bonus + area_bonus + stuck_penalty
    
    def _calculate_proximity_reward(self) -> float:
        """
        Calculate proximity-based reward shaping.
        
        Returns small positive reward for getting closer to food,
        small negative for getting farther. This reward is MUCH smaller
        than food collection reward (100+) to ensure eating food always
        outweighs any proximity penalty.
        """
        if len(self.food_items) == 0:
            # No food available - reset previous distance
            self.prev_food_distance = None
            return 0.0
        
        # Find current distance to nearest food
        min_dist = float('inf')
        for food in self.food_items:
            dist = calculate_distance(self.player_pos, food['pos'])
            min_dist = min(min_dist, dist)
        
        current_distance = min_dist
        
        # If we have a previous distance, compute proximity reward
        if self.prev_food_distance is not None:
            # Reward = change in distance (positive = got closer, negative = got farther)
            # Increased scaling for more meaningful rewards
            distance_change = self.prev_food_distance - current_distance
            proximity_reward = distance_change * 0.5  # Increased from 0.1 to 0.5
            
            # Cap the reward to prevent exploitation (max +/- 2.0 per step)
            proximity_reward = np.clip(proximity_reward, -2.0, 2.0)
        else:
            proximity_reward = 0.0
        
        # Update previous distance for next step
        self.prev_food_distance = current_distance
        
        return proximity_reward
    
    def _calculate_boundary_penalty(self) -> float:
        """
        Calculate a boundary penalty only when literally touching the edges.
        
        Returns a negative reward (penalty) only when actually hitting boundaries.
        """
        penalty = 0.0
        
        x = self.player_pos[0]
        min_x = PLAYER_SIZE // 2
        max_x = WORLD_WIDTH - PLAYER_SIZE // 2
        
        # Only penalize when actually touching the boundaries
        if x <= min_x:
            penalty -= 0.5  # Small penalty for touching left wall
        elif x >= max_x:
            penalty -= 0.5  # Small penalty for touching right wall
        
        return penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with food proximity sensing."""
        # Base state
        obs = np.array([
            self.player_pos[0],      # x position
            self.player_pos[1],      # y position
            self.player_vel[0],      # x velocity
            self.player_vel[1],      # y velocity
            self.player_health,      # health
            self.player_hunger,      # hunger
            self.player_risk         # risk
        ], dtype=np.float32)
        
        # FOOD PROXIMITY SENSING - helps guide exploration
        if len(self.food_items) > 0:
            # Find nearest food
            min_dist = float('inf')
            nearest_food_pos = None
            
            for food in self.food_items:
                dist = calculate_distance(self.player_pos, food['pos'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_food_pos = food['pos']
            
            # Normalized distance (0 = touching, 1 = very far)
            # Using max possible distance in world as normalization
            max_world_dist = np.sqrt(WORLD_WIDTH**2 + WORLD_HEIGHT**2)
            normalized_dist = min(min_dist / max_world_dist, 1.0) if max_world_dist > 0 else 1.0
            
            # Direction vector to nearest food (normalized to [-1, 1])
            if nearest_food_pos:
                dx = (nearest_food_pos[0] - self.player_pos[0]) / WORLD_WIDTH
                dy = (nearest_food_pos[1] - self.player_pos[1]) / WORLD_HEIGHT
            else:
                dx, dy = 0.0, 0.0
        else:
            # No food available - use sentinel values
            normalized_dist = 1.0  # Far away
            dx, dy = 0.0, 0.0  # No direction
        
        # Append food sensing info (distance, direction_x, direction_y)
        food_sensing = np.array([normalized_dist, dx, dy], dtype=np.float32)
        obs = np.concatenate([obs, food_sensing])
        
        return normalize_observation(obs)
    
    def render(self):
        """Render the environment."""
        if self.render_mode != "human" or not PYGAME_AVAILABLE:
            return
        
        self.render_count += 1
        
        # Only render every render_every episodes during training
        if self.render_count % self.render_every != 0:
            return
        
        # Handle keyboard input for HUD controls
        self._handle_input()
        
        # Update visual effects
        self.visual_effects.update()
        
        # Get screen shake offset
        shake_offset = self.visual_effects.get_screen_offset()
        
        # Clear screen
        self.screen.fill((50, 50, 100))  # Dark blue background
        
        # Draw platforms
        for platform in self.platforms:
            color = (100, 100, 100) if platform['type'] == 'ground' else (150, 150, 150)
            pygame.draw.rect(self.screen, color, platform['rect'])
        
        # Draw safe zones
        for safe_zone in self.safe_zones:
            pygame.draw.rect(self.screen, (0, 255, 0, 100), safe_zone['rect'])
        
        # Draw food items
        for food in self.food_items:
            pygame.draw.circle(self.screen, (255, 255, 0), 
                             (int(food['pos'][0]), int(food['pos'][1])), food['size'])
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(enemy['pos'][0]), int(enemy['pos'][1])), enemy['size'])
        
        # Draw player with health-based color
        player_color = get_health_color(self.player_health)
        
        # Flash red if in pain
        if self.pain_flash > 0:
            player_color = (255, 0, 0)
            self.pain_flash -= 1
        
        # Pulse if hungry
        if self.player_hunger > 50:
            self.hunger_pulse += 1
            pulse_factor = 1 + 0.2 * math.sin(self.hunger_pulse * 0.3)
            player_size = int(PLAYER_SIZE * pulse_factor)
        else:
            player_size = PLAYER_SIZE
        
        pygame.draw.circle(self.screen, player_color, 
                         (int(self.player_pos[0] + shake_offset[0]), 
                          int(self.player_pos[1] + shake_offset[1])), player_size)
        
        # Draw visual effects
        self.visual_effects.draw(self.screen)
        
        # Draw enhanced HUD
        self.enhanced_hud.draw(
            self.player_health, self.player_hunger, self.player_risk,
            self.episode_reward, self.episode_count, self.step_count,
            tuple(self.player_pos), self.get_action_meanings()
        )
        
        # Draw level information - show actual remaining food items
        if self.single_level:
            remaining_food = len(self.food_items)
            food_text = self.font.render(f"Food: {remaining_food}/{self.total_food} remaining ({self.food_collected} collected)", True, (255, 255, 255))
            self.screen.blit(food_text, (10, 130))
            
            if self.level_completed:
                completed_text = self.font.render("LEVEL COMPLETED!", True, (0, 255, 0))
                self.screen.blit(completed_text, (10, 150))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _handle_input(self):
        """Handle keyboard input for HUD controls."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.enhanced_hud.toggle_hud()
                elif event.key == pygame.K_m:
                    self.enhanced_hud.toggle_minimap()
                elif event.key == pygame.K_t:
                    self.enhanced_hud.toggle_trajectory()
                elif event.key == pygame.K_a:
                    self.enhanced_hud.show_action_history = not self.enhanced_hud.show_action_history
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'screen'):
            pygame.quit()
    
    def get_action_meanings(self) -> Dict[int, str]:
        """Get human-readable action meanings."""
        return {
            ACTION_LEFT: "Move Left",
            ACTION_RIGHT: "Move Right", 
            ACTION_JUMP: "Jump",
            ACTION_REST: "Rest",
            ACTION_EAT: "Eat"  # Must use ACTION_EAT to consume food
        }
    
    def save_replay_data(self, filename: str = "replay_data.json"):
        """Save replay buffer data to file."""
        if self.render_mode == "human" and hasattr(self, 'replay_buffer'):
            self.replay_buffer.save_trajectories(filename)
            print(f"Replay data saved to {filename}")
    
    def get_replay_buffer(self):
        """Get the replay buffer for analysis."""
        if self.render_mode == "human" and hasattr(self, 'replay_buffer'):
            return self.replay_buffer
        return None
