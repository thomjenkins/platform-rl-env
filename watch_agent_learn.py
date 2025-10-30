#!/usr/bin/env python3
"""
Real-time maze training that you can watch the agent learn.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
import numpy as np
import random
import time

def watch_agent_learn():
    print('🎮 WATCH THE AGENT LEARN IN REAL-TIME!')
    print('👀 You should see a PyGame window with the maze!')
    print('🎯 Agent will learn to navigate, collect food, and reach exit!')
    
    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Watch Agent Learn Maze")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Maze layout
    walls = [
        (0, 0, 800, 20),      # Top wall
        (0, 580, 800, 20),    # Bottom wall
        (0, 0, 20, 600),      # Left wall
        (780, 0, 20, 600),    # Right wall
        (200, 100, 20, 200),  # Vertical wall 1
        (400, 200, 20, 200),  # Vertical wall 2
        (600, 100, 20, 200),  # Vertical wall 3
        (100, 200, 200, 20),  # Horizontal wall 1
        (300, 300, 200, 20),  # Horizontal wall 2
        (500, 200, 200, 20),  # Horizontal wall 3
    ]
    
    # Game state
    player_pos = [50, 550]
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
    ]
    food_collected = 0
    
    # Exit position
    exit_pos = [750, 50]
    
    # Training state
    episode = 0
    step = 0
    total_reward = 0
    episode_rewards = []
    
    # Simple Q-learning table
    q_table = {}
    
    def get_state():
        """Get current state for Q-learning."""
        x = int(player_pos[0] // 50)
        y = int(player_pos[1] // 50)
        return (x, y, food_collected)
    
    def get_q_value(state, action):
        """Get Q-value for state-action pair."""
        if (state, action) not in q_table:
            q_table[(state, action)] = 0
        return q_table[(state, action)]
    
    def update_q_value(state, action, reward, next_state):
        """Update Q-value using Q-learning."""
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        
        current_q = get_q_value(state, action)
        max_next_q = max([get_q_value(next_state, a) for a in range(4)])
        
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        q_table[(state, action)] = new_q
    
    def choose_action(state, epsilon=0.1):
        """Choose action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, 3)  # Random action
        else:
            q_values = [get_q_value(state, a) for a in range(4)]
            return q_values.index(max(q_values))
    
    def calculate_reward():
        """Calculate reward using your original formula."""
        # Your reward function: r_t = u(s_t, a_t; H_t) − c_pain(H_t, a_t) − λ·max(0, H* − H_{t+1}) − α·G_t − β·R_t
        
        # Simple utility based on movement
        utility = 0.1 if abs(player_pos[0] - 50) > 10 or abs(player_pos[1] - 550) > 10 else 0
        
        # Pain cost (simplified)
        pain = 0
        
        # Health loss penalty: λ·max(0, H* − H_{t+1})
        health_loss = max(0, 100 - health)
        health_penalty = 2.0 * health_loss  # λ = 2.0
        
        # Hunger penalty: α·G_t
        hunger_penalty = 0.1 * hunger  # α = 0.1
        
        # Risk penalty: β·R_t
        risk_penalty = 0.2 * risk  # β = 0.2
        
        # Total reward
        reward = utility - pain - health_penalty - hunger_penalty - risk_penalty
        
        return reward
    
    print('🎯 Agent will learn to balance health, hunger, and risk!')
    print('📊 Watch the Q-values and rewards improve!')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current state
        state = get_state()
        
        # Choose action (epsilon decreases over time)
        epsilon = max(0.05, 0.9 - episode * 0.01)
        action = choose_action(state, epsilon)
        
        # Store old state for Q-learning
        old_state = state
        old_pos = player_pos.copy()
        old_health = health
        old_hunger = hunger
        old_risk = risk
        
        # Move player based on action
        if action == 0 and player_pos[0] > 30:  # Left
            player_pos[0] -= 3
        elif action == 1 and player_pos[0] < 750:  # Right
            player_pos[0] += 3
        elif action == 2 and player_pos[1] > 30:  # Up
            player_pos[1] -= 3
        elif action == 3 and player_pos[1] < 550:  # Down
            player_pos[1] += 3
        
        # Check wall collisions
        player_rect = pygame.Rect(player_pos[0] - player_size//2, player_pos[1] - player_size//2, player_size, player_size)
        hit_wall = False
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            if player_rect.colliderect(wall_rect):
                hit_wall = True
                # Move player back
                if action == 0:  # Left
                    player_pos[0] += 3
                elif action == 1:  # Right
                    player_pos[0] -= 3
                elif action == 2:  # Up
                    player_pos[1] += 3
                elif action == 3:  # Down
                    player_pos[1] -= 3
                break
        
        # Update internal states
        hunger += 0.3  # Hunger increases over time
        if hunger > 100:
            hunger = 100
        
        # Check food collection
        for i, food_pos in enumerate(food_positions):
            if food_pos is not None:
                distance = ((player_pos[0] - food_pos[0])**2 + (player_pos[1] - food_pos[1])**2)**0.5
                if distance < 25:
                    food_positions[i] = None
                    food_collected += 1
                    hunger = max(0, hunger - 20)  # Food reduces hunger
                    health = min(100, health + 10)  # Food heals
                    print(f'🍎 Food collected! Total: {food_collected}/5, Hunger: {hunger:.1f}, Health: {health:.1f}')
        
        # Check exit
        distance_to_exit = ((player_pos[0] - exit_pos[0])**2 + (player_pos[1] - exit_pos[1])**2)**0.5
        if distance_to_exit < 30:
            reward = 1000  # Big reward for completion
            print(f'🎉 MAZE COMPLETED! Episode {episode}, Steps: {step}, Total Reward: {total_reward + reward}')
            episode_rewards.append(total_reward + reward)
            episode += 1
            step = 0
            total_reward = 0
            food_collected = 0
            player_pos = [50, 550]
            health = 100
            hunger = 0
            risk = 0
            food_positions = [
                (150, 150),
                (350, 250),
                (550, 150),
                (150, 350),
                (550, 350),
            ]
            continue
        
        # Calculate reward using your formula
        reward = calculate_reward()
        
        # Additional rewards/penalties
        if hit_wall:
            reward -= 10
        reward -= 0.1  # Time penalty
        
        # Update Q-values
        new_state = get_state()
        update_q_value(old_state, action, reward, new_state)
        
        total_reward += reward
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
        
        # Draw HUD
        hud_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Food: {food_collected}/5",
            f"Health: {health:.1f}",
            f"Hunger: {hunger:.1f}",
            f"Risk: {risk:.1f}",
            f"Total Reward: {total_reward:.1f}",
            f"Q-Table Size: {len(q_table)}",
            f"Epsilon: {epsilon:.3f}",
        ]
        
        if episode_rewards:
            avg_reward = sum(episode_rewards[-5:]) / len(episode_rewards[-5:])
            hud_text.append(f"Avg Reward (last 5): {avg_reward:.1f}")
        
        for i, text in enumerate(hud_text):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
        
        # Limit episodes for demo
        if episode >= 15:
            print('✅ Training completed!')
            print(f'📊 Final Q-table size: {len(q_table)}')
            if episode_rewards:
                print(f'📈 Average reward: {sum(episode_rewards) / len(episode_rewards):.1f}')
            break
    
    pygame.quit()
    print('👋 Training finished!')

if __name__ == "__main__":
    watch_agent_learn()
