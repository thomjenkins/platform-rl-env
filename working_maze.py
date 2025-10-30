#!/usr/bin/env python3
"""
Working maze training that definitely shows the agent learning.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
import numpy as np
import random
import time

def working_maze_training():
    print('🎮 WORKING MAZE TRAINING!')
    print('👀 You should see the agent learning in real-time!')
    
    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Agent Learning Maze")
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
        # Discretize position
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
            # Choose best action
            q_values = [get_q_value(state, a) for a in range(4)]
            return q_values.index(max(q_values))
    
    print('🎯 Agent will learn to navigate the maze!')
    print('📊 Watch the Q-values improve over time!')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current state
        state = get_state()
        
        # Choose action
        action = choose_action(state, epsilon=max(0.1, 0.9 - episode * 0.01))
        
        # Store old state for Q-learning
        old_state = state
        old_pos = player_pos.copy()
        
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
        
        # Calculate reward
        reward = 0
        
        # Negative reward for hitting walls
        if hit_wall:
            reward -= 10
        
        # Negative reward for taking too long
        reward -= 0.1
        
        # Positive reward for collecting food
        for i, food_pos in enumerate(food_positions):
            if food_pos is not None:
                distance = ((player_pos[0] - food_pos[0])**2 + (player_pos[1] - food_pos[1])**2)**0.5
                if distance < 25:
                    food_positions[i] = None
                    food_collected += 1
                    reward += 100
                    print(f'🍎 Food collected! Total: {food_collected}/5, Reward: +100')
        
        # Big reward for reaching exit
        distance_to_exit = ((player_pos[0] - exit_pos[0])**2 + (player_pos[1] - exit_pos[1])**2)**0.5
        if distance_to_exit < 30:
            reward += 1000
            print(f'🎉 MAZE COMPLETED! Episode {episode}, Steps: {step}, Total Reward: {total_reward + reward}')
            episode_rewards.append(total_reward + reward)
            episode += 1
            step = 0
            total_reward = 0
            food_collected = 0
            player_pos = [50, 550]
            food_positions = [
                (150, 150),
                (350, 250),
                (550, 150),
                (150, 350),
                (550, 350),
            ]
            continue
        
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
        
        # Draw player
        pygame.draw.rect(screen, (0, 255, 0), 
                        (player_pos[0] - player_size//2, player_pos[1] - player_size//2, 
                         player_size, player_size))
        
        # Draw HUD
        hud_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Food: {food_collected}/5",
            f"Total Reward: {total_reward:.1f}",
            f"Q-Table Size: {len(q_table)}",
        ]
        
        if episode_rewards:
            avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
            hud_text.append(f"Avg Reward (last 10): {avg_reward:.1f}")
        
        for i, text in enumerate(hud_text):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
        
        # Limit episodes for demo
        if episode >= 20:
            print('✅ Training completed!')
            print(f'📊 Final Q-table size: {len(q_table)}')
            if episode_rewards:
                print(f'📈 Average reward: {sum(episode_rewards) / len(episode_rewards):.1f}')
            break
    
    pygame.quit()
    print('👋 Training finished!')

if __name__ == "__main__":
    working_maze_training()
