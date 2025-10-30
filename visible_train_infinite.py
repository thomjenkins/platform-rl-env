#!/usr/bin/env python3
"""
Real-time visible training script for the platform RL environment.
Run this to watch the agent learn in real-time!
"""

import os
import time
import sys

# Set up environment
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

# Import after setting environment
from train_infinite import train_agent

def main():
    print('🎮 INFINITE EPISODE TRAINING - REAL-TIME VISIBLE!')
    print('♾️  Episodes run indefinitely - no death termination!')
    print('👀 You can now watch the agent learn live!')
    print('🎯 Goal: Watch the agent develop strategies')
    print('🔄 Food respawns when all collected - continuous learning!')
    print()
    print('Starting at:', time.strftime('%H:%M:%S'))
    print()
    print('Watch the agent:')
    print('  - Learn to move around the level')
    print('  - Avoid enemies')
    print('  - Collect food items')
    print('  - Use safe zones for healing')
    print('  - Complete the level!')
    print()
    print('Press Ctrl+C to stop when you want.')
    print('=' * 50)
    
    try:
        # Start training with frequent rendering
        train_agent(
            episodes=500,  # Reasonable number for watching
            render_every=1,  # Render every episode so you can see progress
            save_every=25,  # Save frequently
            device='cpu'
        )
    except KeyboardInterrupt:
        print('\n👋 Training stopped by user.')
        print('The agent has learned some strategies!')
    except Exception as e:
        print(f'\n❌ Error during training: {e}')
        print('Try running with: python visible_train.py')

if __name__ == "__main__":
    main()
