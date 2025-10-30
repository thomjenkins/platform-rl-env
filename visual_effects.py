"""
Additional visual features and replay buffer visualizer for the 2D Platform RL Environment.

This module provides enhanced visualization capabilities including trajectory replay,
HUD toggles, and advanced visual effects.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import json
import os
from typing import List, Dict, Tuple, Optional
from utils import WORLD_WIDTH, WORLD_HEIGHT, MAX_HEALTH, MAX_HUNGER, MAX_RISK


class ReplayBuffer:
    """Buffer for storing and replaying agent trajectories."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trajectories = deque(maxlen=max_size)
        self.current_trajectory = []
    
    def add_step(self, obs: np.ndarray, action: int, reward: float, 
                health: float, hunger: float, risk: float, pos: Tuple[float, float]):
        """Add a step to the current trajectory."""
        step_data = {
            'obs': obs.copy(),
            'action': action,
            'reward': reward,
            'health': health,
            'hunger': hunger,
            'risk': risk,
            'pos': pos
        }
        self.current_trajectory.append(step_data)
    
    def end_trajectory(self):
        """End current trajectory and add to buffer."""
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory.copy())
            self.current_trajectory.clear()
    
    def get_trajectory(self, index: int = -1) -> List[Dict]:
        """Get trajectory by index."""
        if not self.trajectories:
            return []
        return self.trajectories[index]
    
    def get_all_trajectories(self) -> List[List[Dict]]:
        """Get all stored trajectories."""
        return list(self.trajectories)
    
    def save_trajectories(self, filename: str):
        """Save trajectories to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_trajectories = []
        for traj in self.trajectories:
            serializable_traj = []
            for step in traj:
                serializable_step = step.copy()
                serializable_step['obs'] = step['obs'].tolist()
                serializable_traj.append(serializable_step)
            serializable_trajectories.append(serializable_traj)
        
        with open(filename, 'w') as f:
            json.dump(serializable_trajectories, f, indent=2)
    
    def load_trajectories(self, filename: str):
        """Load trajectories from JSON file."""
        with open(filename, 'r') as f:
            serializable_trajectories = json.load(f)
        
        self.trajectories.clear()
        for traj in serializable_trajectories:
            trajectory = []
            for step in traj:
                step['obs'] = np.array(step['obs'])
                trajectory.append(step)
            self.trajectories.append(trajectory)


class TrajectoryVisualizer:
    """Visualizer for agent trajectories and internal states."""
    
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Agent Trajectory Analysis')
        
        # Setup subplots
        self.axes[0, 0].set_title('Trajectory Path')
        self.axes[0, 0].set_xlabel('X Position')
        self.axes[0, 0].set_ylabel('Y Position')
        self.axes[0, 0].invert_yaxis()  # Invert Y to match game coordinates
        
        self.axes[0, 1].set_title('Health Over Time')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Health')
        
        self.axes[1, 0].set_title('Hunger Over Time')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Hunger')
        
        self.axes[1, 1].set_title('Risk Over Time')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Risk')
    
    def plot_trajectory(self, trajectory_index: int = -1):
        """Plot a specific trajectory."""
        trajectory = self.replay_buffer.get_trajectory(trajectory_index)
        if not trajectory:
            return
        
        # Extract data
        positions = [step['pos'] for step in trajectory]
        health = [step['health'] for step in trajectory]
        hunger = [step['hunger'] for step in trajectory]
        risk = [step['risk'] for step in trajectory]
        rewards = [step['reward'] for step in trajectory]
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot trajectory path
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions]
        
        # Color trajectory by reward
        scatter = self.axes[0, 0].scatter(x_pos, y_pos, c=rewards, cmap='RdYlGn', s=20)
        self.axes[0, 0].plot(x_pos, y_pos, 'b-', alpha=0.3, linewidth=1)
        self.axes[0, 0].set_title('Trajectory Path (colored by reward)')
        self.axes[0, 0].set_xlabel('X Position')
        self.axes[0, 0].set_ylabel('Y Position')
        self.axes[0, 0].invert_yaxis()
        plt.colorbar(scatter, ax=self.axes[0, 0], label='Reward')
        
        # Plot internal states over time
        steps = range(len(trajectory))
        
        self.axes[0, 1].plot(steps, health, 'g-', linewidth=2, label='Health')
        self.axes[0, 1].set_title('Health Over Time')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Health')
        self.axes[0, 1].set_ylim(0, MAX_HEALTH)
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].plot(steps, hunger, 'orange', linewidth=2, label='Hunger')
        self.axes[1, 0].set_title('Hunger Over Time')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Hunger')
        self.axes[1, 0].set_ylim(0, MAX_HUNGER)
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].plot(steps, risk, 'r-', linewidth=2, label='Risk')
        self.axes[1, 1].set_title('Risk Over Time')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Risk')
        self.axes[1, 1].set_ylim(0, MAX_RISK)
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_trajectories(self, num_trajectories: int = 5):
        """Plot multiple trajectories for comparison."""
        trajectories = self.replay_buffer.get_all_trajectories()
        if not trajectories:
            return
        
        # Take last N trajectories
        recent_trajectories = trajectories[-num_trajectories:]
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot all trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(recent_trajectories)))
        
        for i, trajectory in enumerate(recent_trajectories):
            positions = [step['pos'] for step in trajectory]
            health = [step['health'] for step in trajectory]
            hunger = [step['hunger'] for step in trajectory]
            risk = [step['risk'] for step in trajectory]
            
            x_pos = [pos[0] for pos in positions]
            y_pos = [pos[1] for pos in positions]
            steps = range(len(trajectory))
            
            # Plot trajectory path
            self.axes[0, 0].plot(x_pos, y_pos, color=colors[i], alpha=0.7, linewidth=1)
            
            # Plot internal states
            self.axes[0, 1].plot(steps, health, color=colors[i], alpha=0.7, linewidth=1)
            self.axes[1, 0].plot(steps, hunger, color=colors[i], alpha=0.7, linewidth=1)
            self.axes[1, 1].plot(steps, risk, color=colors[i], alpha=0.7, linewidth=1)
        
        # Set labels and limits
        self.axes[0, 0].set_title(f'Last {len(recent_trajectories)} Trajectories')
        self.axes[0, 0].set_xlabel('X Position')
        self.axes[0, 0].set_ylabel('Y Position')
        self.axes[0, 0].invert_yaxis()
        
        self.axes[0, 1].set_title('Health Over Time')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Health')
        self.axes[0, 1].set_ylim(0, MAX_HEALTH)
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Hunger Over Time')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Hunger')
        self.axes[1, 0].set_ylim(0, MAX_HUNGER)
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Risk Over Time')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Risk')
        self.axes[1, 1].set_ylim(0, MAX_RISK)
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


class EnhancedHUD:
    """Enhanced HUD with toggleable elements and advanced visualizations."""
    
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        self.screen = screen
        self.font = font
        self.show_hud = True
        self.show_minimap = True
        self.show_trajectory = True
        self.show_action_history = True
        
        # Action history
        self.action_history = deque(maxlen=50)
        self.trajectory_points = deque(maxlen=200)
    
    def toggle_hud(self):
        """Toggle HUD visibility."""
        self.show_hud = not self.show_hud
    
    def toggle_minimap(self):
        """Toggle minimap visibility."""
        self.show_minimap = not self.show_minimap
    
    def toggle_trajectory(self):
        """Toggle trajectory visualization."""
        self.show_trajectory = not self.show_trajectory
    
    def add_action(self, action: int, action_names: Dict[int, str]):
        """Add action to history."""
        action_name = action_names.get(action, f"Action {action}")
        # Skip auto-eat actions (hide them from action history)
        if "(auto)" in action_name.lower():
            return
        self.action_history.append((action, action_name))
    
    def add_position(self, pos: Tuple[float, float]):
        """Add position to trajectory."""
        self.trajectory_points.append(pos)
    
    def draw(self, health: float, hunger: float, risk: float, 
             episode_reward: float, episode_count: int, step_count: int,
             player_pos: Tuple[float, float], action_names: Dict[int, str]):
        """Draw the enhanced HUD."""
        if not self.show_hud:
            return
        
        # Main HUD elements
        self._draw_main_hud(health, hunger, risk, episode_reward, episode_count, step_count)
        
        # Minimap
        if self.show_minimap:
            self._draw_minimap(player_pos)
        
        # Trajectory
        if self.show_trajectory:
            self._draw_trajectory()
        
        # Action history
        if self.show_action_history:
            self._draw_action_history(action_names)
        
        # Controls help
        self._draw_controls()
    
    def _draw_main_hud(self, health: float, hunger: float, risk: float,
                      episode_reward: float, episode_count: int, step_count: int):
        """Draw main HUD elements."""
        # Background
        hud_rect = pygame.Rect(5, 5, 300, 120)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), hud_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), hud_rect, 2)
        
        # Health bar
        health_width = int(200 * (health / MAX_HEALTH))
        pygame.draw.rect(self.screen, (255, 0, 0), (15, 15, 200, 15))
        pygame.draw.rect(self.screen, (0, 255, 0), (15, 15, health_width, 15))
        health_text = self.font.render(f"Health: {health:.1f}", True, (255, 255, 255))
        self.screen.blit(health_text, (225, 15))
        
        # Hunger bar
        hunger_width = int(200 * (hunger / MAX_HUNGER))
        pygame.draw.rect(self.screen, (255, 255, 0), (15, 35, hunger_width, 12))
        hunger_text = self.font.render(f"Hunger: {hunger:.1f}", True, (255, 255, 255))
        self.screen.blit(hunger_text, (225, 35))
        
        # Risk bar
        risk_width = int(200 * (risk / MAX_RISK))
        pygame.draw.rect(self.screen, (255, 0, 0), (15, 52, risk_width, 12))
        risk_text = self.font.render(f"Risk: {risk:.1f}", True, (255, 255, 255))
        self.screen.blit(risk_text, (225, 52))
        
        # Episode info
        reward_text = self.font.render(f"Reward: {episode_reward:.2f}", True, (255, 255, 255))
        episode_text = self.font.render(f"Episode: {episode_count}", True, (255, 255, 255))
        step_text = self.font.render(f"Step: {step_count}", True, (255, 255, 255))
        
        self.screen.blit(reward_text, (15, 70))
        self.screen.blit(episode_text, (15, 90))
        self.screen.blit(step_text, (15, 110))
    
    def _draw_minimap(self, player_pos: Tuple[float, float]):
        """Draw minimap in top-right corner."""
        minimap_size = 150
        minimap_rect = pygame.Rect(WORLD_WIDTH - minimap_size - 10, 10, minimap_size, minimap_size)
        
        # Background
        pygame.draw.rect(self.screen, (0, 0, 0, 200), minimap_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), minimap_rect, 2)
        
        # Scale factor
        scale_x = minimap_size / WORLD_WIDTH
        scale_y = minimap_size / WORLD_HEIGHT
        
        # Player position
        player_x = int(minimap_rect.x + player_pos[0] * scale_x)
        player_y = int(minimap_rect.y + player_pos[1] * scale_y)
        pygame.draw.circle(self.screen, (0, 255, 0), (player_x, player_y), 3)
        
        # Title
        title_text = self.font.render("Minimap", True, (255, 255, 255))
        self.screen.blit(title_text, (minimap_rect.x, minimap_rect.y - 25))
    
    def _draw_trajectory(self):
        """Draw trajectory trail."""
        if len(self.trajectory_points) < 2:
            return
        
        # Draw trajectory as connected lines
        points = list(self.trajectory_points)
        for i in range(1, len(points)):
            # Fade older points
            alpha = int(255 * (i / len(points)))
            color = (0, 255, 255, alpha)
            
            start_pos = (int(points[i-1][0]), int(points[i-1][1]))
            end_pos = (int(points[i][0]), int(points[i][1]))
            
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def _draw_action_history(self, action_names: Dict[int, str]):
        """Draw recent action history."""
        if not self.action_history:
            return
        
        # Background
        history_rect = pygame.Rect(WORLD_WIDTH - 200, WORLD_HEIGHT - 150, 190, 140)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), history_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), history_rect, 2)
        
        # Title
        title_text = self.font.render("Recent Actions", True, (255, 255, 255))
        self.screen.blit(title_text, (history_rect.x + 5, history_rect.y + 5))
        
        # Actions
        y_offset = 25
        for i, (action, name) in enumerate(list(self.action_history)[-8:]):
            action_text = self.font.render(f"{i+1}. {name}", True, (255, 255, 255))
            self.screen.blit(action_text, (history_rect.x + 5, history_rect.y + y_offset))
            y_offset += 15
    
    def _draw_controls(self):
        """Draw control help."""
        controls = [
            "H: Toggle HUD",
            "M: Toggle Minimap", 
            "T: Toggle Trajectory",
            "A: Toggle Action History"
        ]
        
        # Background
        controls_rect = pygame.Rect(5, WORLD_HEIGHT - 100, 200, 95)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), controls_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), controls_rect, 2)
        
        # Title
        title_text = self.font.render("Controls", True, (255, 255, 255))
        self.screen.blit(title_text, (controls_rect.x + 5, controls_rect.y + 5))
        
        # Control text
        y_offset = 25
        for control in controls:
            control_text = self.font.render(control, True, (255, 255, 255))
            self.screen.blit(control_text, (controls_rect.x + 5, controls_rect.y + y_offset))
            y_offset += 15


class VisualEffects:
    """Advanced visual effects for the environment."""
    
    def __init__(self):
        self.particles = []
        self.screen_shake = 0
        self.flash_effect = 0
    
    def add_particle(self, pos: Tuple[float, float], color: Tuple[int, int, int], 
                    velocity: Tuple[float, float], lifetime: int = 30):
        """Add a particle effect."""
        self.particles.append({
            'pos': pos,
            'color': color,
            'velocity': velocity,
            'lifetime': lifetime,
            'max_lifetime': lifetime
        })
    
    def add_screen_shake(self, intensity: int = 10):
        """Add screen shake effect."""
        self.screen_shake = max(self.screen_shake, intensity)
    
    def add_flash(self, color: Tuple[int, int, int] = (255, 255, 255), duration: int = 5):
        """Add screen flash effect."""
        self.flash_effect = max(self.flash_effect, duration)
        self.flash_color = color
    
    def update(self):
        """Update all visual effects."""
        # Update particles
        for particle in self.particles[:]:
            particle['pos'] = (particle['pos'][0] + particle['velocity'][0],
                             particle['pos'][1] + particle['velocity'][1])
            particle['lifetime'] -= 1
            
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)
        
        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1
        
        # Update flash
        if self.flash_effect > 0:
            self.flash_effect -= 1
    
    def draw(self, screen: pygame.Surface):
        """Draw all visual effects."""
        # Draw particles
        for particle in self.particles:
            alpha = int(255 * (particle['lifetime'] / particle['max_lifetime']))
            color = (*particle['color'], alpha)
            
            pos = (int(particle['pos'][0]), int(particle['pos'][1]))
            pygame.draw.circle(screen, particle['color'], pos, 3)
        
        # Draw flash effect
        if self.flash_effect > 0:
            alpha = int(255 * (self.flash_effect / 5))
            flash_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
            flash_surface.set_alpha(alpha)
            flash_surface.fill(self.flash_color)
            screen.blit(flash_surface, (0, 0))
    
    def get_screen_offset(self) -> Tuple[int, int]:
        """Get screen shake offset."""
        if self.screen_shake > 0:
            offset_x = np.random.randint(-self.screen_shake, self.screen_shake + 1)
            offset_y = np.random.randint(-self.screen_shake, self.screen_shake + 1)
            return (offset_x, offset_y)
        return (0, 0)
