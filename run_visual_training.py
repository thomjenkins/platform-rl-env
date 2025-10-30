#!/usr/bin/env python3
"""
Visual training script that should show the PyGame window.
"""

import os
import sys

# Set up environment for PyGame
os.environ['SDL_VIDEODRIVER'] = 'cocoa'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def main():
    print('🎮 Starting VISUAL TRAINING!')
    print('👀 You should see a PyGame window with the game!')
    print('🎯 Watch the agent learn in real-time!')
    print()
    
    try:
        # Import and run training
        from train import train_agent
        
        print('Starting training with visual display...')
        train_agent(
            episodes=1000,
            render_every=1,  # Render every episode
            save_every=25,   # Save every 25 episodes
            device='cpu'
        )
        
    except Exception as e:
        print(f'Error: {e}')
        print('Trying alternative approach...')
        
        # Try with different PyGame settings
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        try:
            from train import train_agent
            train_agent(episodes=100, render_every=1, save_every=10, device='cpu')
        except Exception as e2:
            print(f'Alternative also failed: {e2}')

if __name__ == "__main__":
    main()
