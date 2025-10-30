"""
Demo script for the 2D Platform RL Environment with enhanced visual features.

This script demonstrates the environment's capabilities including:
- Interactive HUD controls
- Visual effects and particle systems
- Replay buffer visualization
- Trajectory analysis
"""

import pygame
import numpy as np
import time
import random
from env_platform import Platform2DEnv
from visual_effects import TrajectoryVisualizer
from utils import ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_REST, ACTION_EAT


def manual_play_demo():
    """Demo where user can manually control the agent."""
    print("Manual Play Demo")
    print("Controls:")
    print("  Arrow Keys: Move left/right")
    print("  Space: Jump")
    print("  R: Rest")
    print("  E: Eat")
    print("  H: Toggle HUD")
    print("  M: Toggle Minimap")
    print("  T: Toggle Trajectory")
    print("  A: Toggle Action History")
    print("  ESC: Quit")
    print("-" * 50)
    
    env = Platform2DEnv(render_mode="human", render_every=1)
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = ACTION_REST  # Default action
        
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    action = ACTION_LEFT
                elif event.key == pygame.K_RIGHT:
                    action = ACTION_RIGHT
                elif event.key == pygame.K_SPACE:
                    action = ACTION_JUMP
                elif event.key == pygame.K_r:
                    action = ACTION_REST
                elif event.key == pygame.K_e:
                    action = ACTION_EAT
                elif event.key == pygame.K_h:
                    env.enhanced_hud.toggle_hud()
                elif event.key == pygame.K_m:
                    env.enhanced_hud.toggle_minimap()
                elif event.key == pygame.K_t:
                    env.enhanced_hud.toggle_trajectory()
                elif event.key == pygame.K_a:
                    env.enhanced_hud.show_action_history = not env.enhanced_hud.show_action_history
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Print info
        if terminated or truncated:
            print(f"Episode ended! Final stats:")
            print(f"  Health: {info['health']:.1f}")
            print(f"  Hunger: {info['hunger']:.1f}")
            print(f"  Risk: {info['risk']:.1f}")
            print(f"  Total Reward: {info['episode_reward']:.2f}")
            print(f"  Steps: {info['step']}")
            
            # Save replay data
            env.save_replay_data("manual_demo_replay.json")
            
            # Ask if user wants to continue
            print("\nPress any key to start new episode, or ESC to quit...")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        else:
                            obs, info = env.reset()
                            waiting = False
        
        clock.tick(60)  # 60 FPS
    
    env.close()


def random_agent_demo():
    """Demo with a random agent to show visual effects."""
    print("Random Agent Demo")
    print("Watch the random agent explore the environment...")
    print("Press ESC to quit")
    print("-" * 50)
    
    env = Platform2DEnv(render_mode="human", render_every=1)
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Random action
        action = random.randint(0, 4)
        
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    env.enhanced_hud.toggle_hud()
                elif event.key == pygame.K_m:
                    env.enhanced_hud.toggle_minimap()
                elif event.key == pygame.K_t:
                    env.enhanced_hud.toggle_trajectory()
                elif event.key == pygame.K_a:
                    env.enhanced_hud.show_action_history = not env.enhanced_hud.show_action_history
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Reset if episode ends
        if terminated or truncated:
            print(f"Episode {env.episode_count} ended!")
            print(f"  Health: {info['health']:.1f}, Hunger: {info['hunger']:.1f}, Risk: {info['risk']:.1f}")
            print(f"  Reward: {info['episode_reward']:.2f}, Steps: {info['step']}")
            
            # Save replay data every 5 episodes
            if env.episode_count % 5 == 0:
                env.save_replay_data(f"random_demo_replay_{env.episode_count}.json")
            
            obs, info = env.reset()
        
        clock.tick(30)  # 30 FPS for smoother demo
    
    env.close()


def trajectory_analysis_demo():
    """Demo showing trajectory analysis capabilities."""
    print("Trajectory Analysis Demo")
    print("This will run a few episodes and then show trajectory analysis...")
    print("-" * 50)
    
    env = Platform2DEnv(render_mode="human", render_every=1)
    
    # Run a few episodes to collect data
    for episode in range(3):
        print(f"Running episode {episode + 1}...")
        obs, info = env.reset()
        
        for step in range(500):  # Max 500 steps per episode
            action = random.randint(0, 4)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 50 == 0:  # Render every 50 steps
                env.render()
                time.sleep(0.1)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: Reward={info['episode_reward']:.2f}")
    
    # Save replay data
    env.save_replay_data("trajectory_analysis_replay.json")
    
    # Show trajectory analysis
    replay_buffer = env.get_replay_buffer()
    if replay_buffer and replay_buffer.trajectories:
        print("\nAnalyzing trajectories...")
        visualizer = TrajectoryVisualizer(replay_buffer)
        
        # Plot individual trajectory
        print("Plotting individual trajectory...")
        visualizer.plot_trajectory(-1)  # Last trajectory
        
        # Plot multiple trajectories
        print("Plotting multiple trajectories...")
        visualizer.plot_multiple_trajectories(3)
        
        print("Trajectory analysis complete!")
    
    env.close()


def main():
    """Main demo function."""
    print("2D Platform RL Environment - Enhanced Features Demo")
    print("=" * 60)
    print("Choose a demo:")
    print("1. Manual Play (you control the agent)")
    print("2. Random Agent (watch random exploration)")
    print("3. Trajectory Analysis (collect and analyze data)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                manual_play_demo()
                break
            elif choice == "2":
                random_agent_demo()
                break
            elif choice == "3":
                trajectory_analysis_demo()
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
