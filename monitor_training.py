#!/usr/bin/env python3
"""
Real-time training monitor for the platform RL environment.
Shows live progress, statistics, and learning curves.
"""

import os
import json
import time
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def get_latest_log():
    """Get the most recent training log file."""
    log_files = glob.glob('logs/training_logs_*.json')
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def load_training_data(log_file):
    """Load training data from log file."""
    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except:
        return None

def print_training_stats(data):
    """Print current training statistics."""
    if not data:
        print("No training data available yet...")
        return
    
    episodes = len(data['episode_rewards'])
    if episodes == 0:
        print("No episodes completed yet...")
        return
    
    # Recent performance
    recent_rewards = data['episode_rewards'][-10:]
    recent_lengths = data['episode_lengths'][-10:]
    recent_health = data['episode_health'][-10:]
    
    # Calculate statistics
    avg_reward = np.mean(recent_rewards)
    avg_length = np.mean(recent_lengths)
    avg_health = np.mean(recent_health)
    
    # Best performance
    best_reward = max(data['episode_rewards'])
    best_episode = data['episode_rewards'].index(best_reward) + 1
    
    print(f"\n📊 TRAINING PROGRESS - Episode {episodes}")
    print(f"   Recent Avg Reward: {avg_reward:.2f}")
    print(f"   Recent Avg Length: {avg_length:.1f} steps")
    print(f"   Recent Avg Health: {avg_health:.1f}")
    print(f"   Best Reward: {best_reward:.2f} (Episode {best_episode})")
    
    # Check for level completions
    if 'level_completions' in data:
        completions = data['level_completions']
        completion_rate = completions.count(True) / len(completions) * 100
        print(f"   Level Completion Rate: {completion_rate:.1f}%")
    
    # Training updates
    if 'policy_losses' in data:
        updates = len(data['policy_losses'])
        print(f"   Training Updates: {updates}")

def plot_learning_curve(data, save_path='learning_curve.png'):
    """Plot and save learning curve."""
    if not data or len(data['episode_rewards']) < 10:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    episodes = range(1, len(data['episode_rewards']) + 1)
    
    # Episode rewards
    ax1.plot(episodes, data['episode_rewards'], alpha=0.6, color='blue')
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Episode lengths
    ax2.plot(episodes, data['episode_lengths'], alpha=0.6, color='green')
    ax2.set_title('Episode Lengths Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    # Average health
    ax3.plot(episodes, data['episode_health'], alpha=0.6, color='red')
    ax3.set_title('Average Health Over Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Health')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main monitoring loop."""
    print("🔍 Starting Training Monitor...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    last_episode_count = 0
    
    try:
        while True:
            log_file = get_latest_log()
            if log_file:
                data = load_training_data(log_file)
                if data and len(data['episode_rewards']) > last_episode_count:
                    print_training_stats(data)
                    
                    # Plot learning curve every 10 episodes
                    if len(data['episode_rewards']) % 10 == 0:
                        plot_learning_curve(data)
                        print(f"   📈 Learning curve saved to learning_curve.png")
                    
                    last_episode_count = len(data['episode_rewards'])
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
        print("Final learning curve saved to learning_curve.png")

if __name__ == "__main__":
    main()
