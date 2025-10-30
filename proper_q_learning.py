#!/usr/bin/env python3
"""
Proper Q-learning agent that learns state-action-reward mappings.
The agent learns: "In state S, action A leads to reward R and new state S'"
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
import numpy as np
import random
import time
import json

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table: state -> action -> Q-value
        self.q_table = {}
        
        # Experience buffer for learning
        self.experience_buffer = []
        
        # Statistics
        self.total_steps = 0
        self.total_rewards = 0
        self.episode_rewards = []
        
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        # Discretize continuous values for Q-table
        x = int(state[0] // 20)  # Discretize position
        y = int(state[1] // 20)
        health_bucket = int(state[2] // 20)  # Discretize health
        hunger_bucket = int(state[3] // 10)  # Discretize hunger
        risk_bucket = int(state[4] // 10)   # Discretize risk
        food_collected = int(state[5])      # Food count
        
        return (x, y, health_bucket, hunger_bucket, risk_bucket, food_collected)
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        return self.q_table[state_key][action]
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            q_values = self.q_table[state_key]
            return q_values.index(max(q_values))
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning algorithm."""
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state - no future rewards
            target_q = reward
        else:
            # Non-terminal state - add discounted future reward
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.action_size)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (target_q - current_q)
        
        # Store in Q-table
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to buffer for potential replay."""
        self.experience_buffer.append((state, action, reward, next_state, done))
        if len(self.experience_buffer) > 1000:  # Keep buffer size manageable
            self.experience_buffer.pop(0)
    
    def get_stats(self):
        """Get learning statistics."""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'avg_reward': sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'total_episodes': len(self.episode_rewards)
        }

def watch_agent_learn():
    print('🧠 PROPER Q-LEARNING AGENT - WATCH IT LEARN!')
    print('📚 Agent learns: State S + Action A → Reward R + Next State S\'')
    print('🎯 Using your reward function with health, hunger, risk!')
    
    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("Q-Learning Agent Learning Maze")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)
    big_font = pygame.font.Font(None, 36)
    
    # Maze layout
    walls = [
        (0, 0, 900, 20),      # Top wall
        (0, 680, 900, 20),    # Bottom wall
        (0, 0, 20, 700),      # Left wall
        (880, 0, 20, 700),    # Right wall
        (200, 100, 20, 200),  # Vertical wall 1
        (400, 200, 20, 200),  # Vertical wall 2
        (600, 100, 20, 200),  # Vertical wall 3
        (100, 200, 200, 20),  # Horizontal wall 1
        (300, 300, 200, 20),  # Horizontal wall 2
        (500, 200, 200, 20),  # Horizontal wall 3
        (200, 400, 200, 20),  # Additional wall
    ]
    
    # Initialize agent
    agent = QLearningAgent(state_size=6, action_size=4)  # [x, y, health, hunger, risk, food_collected]
    
    # Game state
    player_pos = [50, 650]
    player_size = 20
    health = 100
    hunger = 0
    risk = 0
    
    # Food positions
    food_positions = [
        (150, 150),
        (350, 250),
        (550, 150),
        (150, 350),
        (550, 350),
        (300, 500),
    ]
    food_collected = 0
    
    # Exit position
    exit_pos = [850, 50]
    
    # Training state
    episode = 0
    step = 0
    total_reward = 0
    episode_rewards = []
    
    # Previous state for Q-learning
    prev_state = None
    prev_action = None
    
    def get_current_state():
        """Get current state vector."""
        return np.array([
            player_pos[0],      # x position
            player_pos[1],      # y position
            health,             # health
            hunger,             # hunger
            risk,               # risk
            food_collected      # food collected
        ])
    
    def calculate_reward():
        """Calculate reward using your original formula."""
        # Your reward function: r_t = u(s_t, a_t; H_t) − c_pain(H_t, a_t) − λ·max(0, H* − H_{t+1}) − α·G_t − β·R_t
        
        # Utility based on movement (encourage exploration)
        movement_utility = 0.1
        
        # Pain cost (simplified - no pain for now)
        pain_cost = 0
        
        # Health loss penalty: λ·max(0, H* − H_{t+1})
        health_loss = max(0, 100 - health)
        health_penalty = 2.0 * health_loss  # λ = 2.0
        
        # Hunger penalty: α·G_t
        hunger_penalty = 0.1 * hunger  # α = 0.1
        
        # Risk penalty: β·R_t
        risk_penalty = 0.2 * risk  # β = 0.2
        
        # Base reward
        reward = movement_utility - pain_cost - health_penalty - hunger_penalty - risk_penalty
        
        return reward
    
    def reset_episode():
        """Reset for new episode."""
        nonlocal player_pos, health, hunger, risk, food_collected, step, total_reward, prev_state, prev_action
        
        player_pos = [50, 650]
        health = 100
        hunger = 0
        risk = 0
        food_collected = 0
        step = 0
        total_reward = 0
        prev_state = None
        prev_action = None
        
        # Reset food
        food_positions[:] = [
            (150, 150),
            (350, 250),
            (550, 150),
            (150, 350),
            (550, 350),
            (300, 500),
        ]
    
    print('🎯 Agent will learn proper state-action-reward mappings!')
    print('📊 Watch Q-values improve as agent explores!')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current state
        current_state = get_current_state()
        
        # Choose action
        action = agent.choose_action(current_state)
        
        # Store previous state-action for Q-learning update
        if prev_state is not None and prev_action is not None:
            # Update Q-value based on previous experience
            agent.update_q_value(prev_state, prev_action, total_reward, current_state, False)
        
        # Move player based on action
        old_pos = player_pos.copy()
        if action == 0 and player_pos[0] > 30:  # Left
            player_pos[0] -= 5
        elif action == 1 and player_pos[0] < 850:  # Right
            player_pos[0] += 5
        elif action == 2 and player_pos[1] > 30:  # Up
            player_pos[1] -= 5
        elif action == 3 and player_pos[1] < 650:  # Down
            player_pos[1] += 5
        
        # Check wall collisions
        player_rect = pygame.Rect(player_pos[0] - player_size//2, player_pos[1] - player_size//2, player_size, player_size)
        hit_wall = False
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            if player_rect.colliderect(wall_rect):
                hit_wall = True
                # Move player back
                player_pos = old_pos
                break
        
        # Update internal states
        hunger += 0.3  # Hunger increases over time
        if hunger > 100:
            hunger = 100
        
        # Check food collection
        food_reward = 0
        for i, food_pos in enumerate(food_positions):
            if food_pos is not None:
                distance = ((player_pos[0] - food_pos[0])**2 + (player_pos[1] - food_pos[1])**2)**0.5
                if distance < 25:
                    food_positions[i] = None
                    food_collected += 1
                    hunger = max(0, hunger - 20)  # Food reduces hunger
                    health = min(100, health + 10)  # Food heals
                    food_reward = 50  # Big reward for food
                    print(f'🍎 Food collected! Total: {food_collected}/6, Hunger: {hunger:.1f}, Health: {health:.1f}')
        
        # Check exit
        distance_to_exit = ((player_pos[0] - exit_pos[0])**2 + (player_pos[1] - exit_pos[1])**2)**0.5
        completion_reward = 0
        done = False
        
        if distance_to_exit < 30:
            completion_reward = 1000  # Big reward for completion
            done = True
            print(f'🎉 MAZE COMPLETED! Episode {episode}, Steps: {step}, Total Reward: {total_reward + completion_reward}')
            episode_rewards.append(total_reward + completion_reward)
            
            # Final Q-learning update for terminal state
            if prev_state is not None and prev_action is not None:
                agent.update_q_value(prev_state, prev_action, completion_reward, current_state, True)
            
            episode += 1
            reset_episode()
            continue
        
        # Calculate reward using your formula
        base_reward = calculate_reward()
        
        # Additional rewards/penalties
        if hit_wall:
            base_reward -= 10
        
        # Time penalty
        base_reward -= 0.1
        
        # Total reward for this step
        step_reward = base_reward + food_reward + completion_reward
        total_reward += step_reward
        
        # Store experience
        agent.add_experience(current_state, action, step_reward, get_current_state(), done)
        
        # Update agent statistics
        agent.total_steps += 1
        agent.total_rewards += step_reward
        
        # Update previous state-action for next iteration
        prev_state = current_state.copy()
        prev_action = action
        
        step += 1
        
        # Render
        screen.fill((20, 20, 60))  # Dark blue background
        
        # Draw walls
        for wall in walls:
            pygame.draw.rect(screen, (100, 100, 100), wall)
        
        # Draw food
        for food_pos in food_positions:
            if food_pos is not None:
                pygame.draw.circle(screen, (255, 255, 0), food_pos, 8)
        
        # Draw exit
        pygame.draw.circle(screen, (0, 255, 0), exit_pos, 15)
        pygame.draw.circle(screen, (255, 255, 255), exit_pos, 15, 3)
        
        # Draw player (color changes with health)
        player_color = (int(255 * (100 - health) / 100), int(255 * health / 100), 0)
        pygame.draw.rect(screen, player_color, 
                        (player_pos[0] - player_size//2, player_pos[1] - player_size//2, 
                         player_size, player_size))
        
        # Draw Q-values for current state (if available)
        state_key = agent.get_state_key(current_state)
        if state_key in agent.q_table:
            q_values = agent.q_table[state_key]
            max_q = max(q_values)
            for i, q_val in enumerate(q_values):
                color_intensity = int(255 * (q_val - min(q_values)) / (max_q - min(q_values) + 0.001))
                color = (color_intensity, 0, 255 - color_intensity)
                pygame.draw.rect(screen, color, (10 + i * 15, 200, 12, 12))
        
        # Draw HUD
        stats = agent.get_stats()
        hud_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Food: {food_collected}/6",
            f"Health: {health:.1f}",
            f"Hunger: {hunger:.1f}",
            f"Risk: {risk:.1f}",
            f"Step Reward: {step_reward:.2f}",
            f"Total Reward: {total_reward:.1f}",
            f"Q-Table Size: {stats['q_table_size']}",
            f"Epsilon: {stats['epsilon']:.3f}",
            f"Learning Rate: {agent.learning_rate:.3f}",
            f"Discount Factor: {agent.discount_factor:.3f}",
        ]
        
        if episode_rewards:
            avg_reward = sum(episode_rewards[-5:]) / len(episode_rewards[-5:])
            hud_text.append(f"Avg Reward (last 5): {avg_reward:.1f}")
        
        # Show current Q-values
        if state_key in agent.q_table:
            q_values = agent.q_table[state_key]
            hud_text.append(f"Q-values: L:{q_values[0]:.2f} R:{q_values[1]:.2f} U:{q_values[2]:.2f} D:{q_values[3]:.2f}")
        
        for i, text in enumerate(hud_text):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 20))
        
        # Draw learning progress
        if len(episode_rewards) > 1:
            pygame.draw.line(screen, (255, 255, 0), (700, 100), (700, 200), 2)
            for i in range(1, min(len(episode_rewards), 50)):
                x1 = 700 + i * 2
                x2 = 700 + (i-1) * 2
                y1 = 200 - int(episode_rewards[-i] / 10)
                y2 = 200 - int(episode_rewards[-(i+1)] / 10)
                pygame.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), 1)
        
        pygame.display.flip()
        clock.tick(30)  # Slower for better observation
        
        # Limit episodes for demo
        if episode >= 20:
            print('✅ Training completed!')
            print(f'📊 Final Q-table size: {stats["q_table_size"]}')
            print(f'📈 Average reward: {stats["avg_reward"]:.1f}')
            print(f'🎯 Total episodes: {stats["total_episodes"]}')
            break
    
    pygame.quit()
    print('👋 Training finished!')
    print('🧠 The agent learned proper state-action-reward mappings!')

if __name__ == "__main__":
    watch_agent_learn()
