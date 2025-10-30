#!/usr/bin/env python3
"""
Simple maze training script that should definitely show the window.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
import numpy as np
import random
import time

def simple_maze_training():
    print('🎮 Starting Simple Maze Training!')
    print('👀 You should see a PyGame window with a maze!')
    
    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Simple Maze RL Training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Simple maze layout
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
    
    # Player position
    player_pos = [50, 550]
    player_size = 20
    
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
    
    episode = 0
    step = 0
    
    print('🎯 Goal: Collect all food and reach the exit!')
    print('🎮 Use arrow keys to move (if you want to test manually)')
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simple AI: random movement
        action = random.randint(0, 3)  # 0=left, 1=right, 2=up, 3=down
        
        # Move player based on action
        if action == 0 and player_pos[0] > 30:  # Left
            player_pos[0] -= 2
        elif action == 1 and player_pos[0] < 750:  # Right
            player_pos[0] += 2
        elif action == 2 and player_pos[1] > 30:  # Up
            player_pos[1] -= 2
        elif action == 3 and player_pos[1] < 550:  # Down
            player_pos[1] += 2
        
        # Check wall collisions
        player_rect = pygame.Rect(player_pos[0] - player_size//2, player_pos[1] - player_size//2, player_size, player_size)
        for wall in walls:
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            if player_rect.colliderect(wall_rect):
                # Move player back
                if action == 0:  # Left
                    player_pos[0] += 2
                elif action == 1:  # Right
                    player_pos[0] -= 2
                elif action == 2:  # Up
                    player_pos[1] += 2
                elif action == 3:  # Down
                    player_pos[1] -= 2
                break
        
        # Check food collection
        for i, food_pos in enumerate(food_positions):
            if food_pos is not None:
                distance = ((player_pos[0] - food_pos[0])**2 + (player_pos[1] - food_pos[1])**2)**0.5
                if distance < 25:
                    food_positions[i] = None
                    food_collected += 1
                    print(f'🍎 Food collected! Total: {food_collected}/5')
        
        # Check exit
        distance_to_exit = ((player_pos[0] - exit_pos[0])**2 + (player_pos[1] - exit_pos[1])**2)**0.5
        if distance_to_exit < 30:
            print(f'🎉 MAZE COMPLETED! Episode {episode}, Steps: {step}')
            episode += 1
            step = 0
            food_collected = 0
            player_pos = [50, 550]
            food_positions = [
                (150, 150),
                (350, 250),
                (550, 150),
                (150, 350),
                (550, 350),
            ]
        
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
            f"Position: ({player_pos[0]}, {player_pos[1]})",
        ]
        
        for i, text in enumerate(hud_text):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
        
        # Limit episodes for demo
        if episode >= 10:
            print('✅ Demo completed!')
            break
    
    pygame.quit()
    print('👋 Training finished!')

if __name__ == "__main__":
    simple_maze_training()
