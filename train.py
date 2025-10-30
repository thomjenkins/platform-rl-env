"""
Training script for the 2D Platform RL Environment.

This script implements the training loop with PPO agent, logging, and visualization.
"""

import os
import time
import argparse
import numpy as np
# import matplotlib.pyplot as plt  # Disabled to prevent blank windows
from collections import deque
import torch
import json
from datetime import datetime

from env_platform import Platform2DEnv
from agent import AdaptivePPOAgent
from utils import OBS_DIM, NUM_ACTIONS, MAX_HEALTH, MAX_HUNGER, MAX_RISK

# Import environment variants
try:
    from env_platform_sensing import Platform2DEnvSensing, OBS_DIM_SENSING
    SENSING_AVAILABLE = True
except ImportError:
    SENSING_AVAILABLE = False

try:
    from env_platform_proximity_reward import Platform2DEnvProximityReward
    PROXIMITY_REWARD_AVAILABLE = True
except ImportError:
    PROXIMITY_REWARD_AVAILABLE = False

try:
    from env_platform_kindest import Platform2DEnvKindest
    KINDEST_AVAILABLE = True
except ImportError:
    KINDEST_AVAILABLE = False


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_health = []
        self.episode_hunger = []
        self.episode_risk = []
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        # Rolling averages
        self.reward_window = deque(maxlen=100)
        self.length_window = deque(maxlen=100)
        self.health_window = deque(maxlen=100)
        self.hunger_window = deque(maxlen=100)
        self.risk_window = deque(maxlen=100)
        
        # Setup matplotlib
        # plt.ion()  # Disabled to prevent blank windows
        # self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))  # Disabled
        # self.fig.suptitle('2D Platform RL Training Progress')  # Disabled
        
        # Initialize plots
        # self._init_plots()  # Disabled
    
    def _init_plots(self):
        """Initialize matplotlib plots."""
        titles = [
            'Episode Rewards', 'Episode Lengths', 'Average Health',
            'Average Hunger', 'Average Risk', 'Training Losses'
        ]
        
        for i, (ax, title) in enumerate(zip(self.axes.flat, titles)):
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.grid(True)
            
            if i < 5:  # Episode metrics
                ax.set_ylabel('Value')
            else:  # Training losses
                ax.set_ylabel('Loss')
    
    def log_episode(self, reward: float, length: int, health: float, 
                   hunger: float, risk: float):
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_health.append(health)
        self.episode_hunger.append(hunger)
        self.episode_risk.append(risk)
        
        # Update rolling averages
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.health_window.append(health)
        self.hunger_window.append(hunger)
        self.risk_window.append(risk)
    
    def log_training(self, losses: dict):
        """Log training losses."""
        self.policy_losses.append(losses.get('policy_loss', 0))
        self.value_losses.append(losses.get('value_loss', 0))
        self.entropy_losses.append(losses.get('entropy_loss', 0))
    
    def update_plots(self, episode: int):
        """Update matplotlib plots."""
        if episode % 10 == 0:  # Update every 10 episodes
            # Clear axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot episode rewards
            self.axes[0, 0].plot(self.episode_rewards, 'b-', alpha=0.7)
            if len(self.reward_window) > 0:
                avg_rewards = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.axes[0, 0].plot(range(9, len(avg_rewards) + 9), avg_rewards, 'r-', linewidth=2)
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].grid(True)
            
            # Plot episode lengths
            self.axes[0, 1].plot(self.episode_lengths, 'g-', alpha=0.7)
            if len(self.length_window) > 0:
                avg_lengths = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
                self.axes[0, 1].plot(range(9, len(avg_lengths) + 9), avg_lengths, 'r-', linewidth=2)
            self.axes[0, 1].set_title('Episode Lengths')
            self.axes[0, 1].set_ylabel('Steps')
            self.axes[0, 1].grid(True)
            
            # Plot average health
            self.axes[0, 2].plot(self.episode_health, 'purple', alpha=0.7)
            if len(self.health_window) > 0:
                avg_health = np.convolve(self.episode_health, np.ones(10)/10, mode='valid')
                self.axes[0, 2].plot(range(9, len(avg_health) + 9), avg_health, 'r-', linewidth=2)
            self.axes[0, 2].set_title('Average Health')
            self.axes[0, 2].set_ylabel('Health')
            self.axes[0, 2].set_ylim(0, MAX_HEALTH)
            self.axes[0, 2].grid(True)
            
            # Plot average hunger
            self.axes[1, 0].plot(self.episode_hunger, 'orange', alpha=0.7)
            if len(self.hunger_window) > 0:
                avg_hunger = np.convolve(self.episode_hunger, np.ones(10)/10, mode='valid')
                self.axes[1, 0].plot(range(9, len(avg_hunger) + 9), avg_hunger, 'r-', linewidth=2)
            self.axes[1, 0].set_title('Average Hunger')
            self.axes[1, 0].set_ylabel('Hunger')
            self.axes[1, 0].set_ylim(0, MAX_HUNGER)
            self.axes[1, 0].grid(True)
            
            # Plot average risk
            self.axes[1, 1].plot(self.episode_risk, 'red', alpha=0.7)
            if len(self.risk_window) > 0:
                avg_risk = np.convolve(self.episode_risk, np.ones(10)/10, mode='valid')
                self.axes[1, 1].plot(range(9, len(avg_risk) + 9), avg_risk, 'r-', linewidth=2)
            self.axes[1, 1].set_title('Average Risk')
            self.axes[1, 1].set_ylabel('Risk')
            self.axes[1, 1].set_ylim(0, MAX_RISK)
            self.axes[1, 1].grid(True)
            
            # Plot training losses
            if len(self.policy_losses) > 0:
                self.axes[1, 2].plot(self.policy_losses, 'b-', label='Policy Loss')
                self.axes[1, 2].plot(self.value_losses, 'g-', label='Value Loss')
                self.axes[1, 2].plot(self.entropy_losses, 'r-', label='Entropy Loss')
                self.axes[1, 2].legend()
            self.axes[1, 2].set_title('Training Losses')
            self.axes[1, 2].set_ylabel('Loss')
            self.axes[1, 2].grid(True)
            
            # plt.tight_layout()  # Disabled to prevent blank windows
            # plt.draw()  # Disabled to prevent blank windows  
            # plt.pause(0.01)  # Disabled to prevent blank windows
    
    def save_logs(self, filename: str = None):
        """Save logs to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_logs_{timestamp}.json"
        
        logs = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_health': self.episode_health,
            'episode_hunger': self.episode_hunger,
            'episode_risk': self.episode_risk,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'rolling_averages': {
                'reward': list(self.reward_window),
                'length': list(self.length_window),
                'health': list(self.health_window),
                'hunger': list(self.hunger_window),
                'risk': list(self.risk_window)
            }
        }
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Logs saved to {filepath}")
    
    def print_stats(self, episode: int):
        """Print current training statistics."""
        if len(self.reward_window) > 0:
            avg_reward = np.mean(self.reward_window)
            avg_length = np.mean(self.length_window)
            avg_health = np.mean(self.health_window)
            avg_hunger = np.mean(self.hunger_window)
            avg_risk = np.mean(self.risk_window)
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Avg Health: {avg_health:5.1f} | "
                  f"Avg Hunger: {avg_hunger:5.1f} | "
                  f"Avg Risk: {avg_risk:5.1f}")


def train_agent(
    episodes: int = 1000,
    max_steps: int = 1000,
    render_every: int = 10,
    save_every: int = 100,
    log_dir: str = "logs",
    model_dir: str = "models",
    device: str = "cpu",
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    update_frequency: int = 2048,
    agent_name: str = None,
    environment: str = "base",
    load_agent: str = None
):
    """
    Train the PPO agent.
    
    Args:
        agent_name: Name for this agent (e.g., "Bob", "Berta"). If None, uses timestamp.
        environment: Environment identifier (e.g., "base", "sensing", "proximity_reward")
        load_agent: Path to previous agent to continue training (transfers learning history)
    """
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate agent name if not provided
    if agent_name is None:
        import datetime
        agent_name = f"agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🤖 Agent: {agent_name}")
    print(f"🌍 Environment: {environment}")
    
    # Initialize environment based on environment identifier
    # Use None for render_mode to disable rendering for speed (or "human" to see it)
    # Only disable rendering if render_every is very high (e.g., > 100) to save computation
    render_mode = None if render_every > 100 else "human"  # Render if render_every <= 100
    
    # Set env_render_every=1 since we control episode-level rendering from training loop
    env_render_every = 1 if render_mode == "human" else render_every
    
    if environment == "sensing" and SENSING_AVAILABLE:
        env = Platform2DEnvSensing(render_mode=render_mode, render_every=env_render_every)
        obs_dim = OBS_DIM_SENSING
        print(f"✅ Using sensing environment (obs_dim={obs_dim})")
    elif environment == "proximity_reward" and PROXIMITY_REWARD_AVAILABLE:
        env = Platform2DEnvProximityReward(render_mode=render_mode, render_every=env_render_every)
        obs_dim = OBS_DIM_SENSING  # Same as sensing
        print(f"✅ Using proximity_reward environment (obs_dim={obs_dim})")
    elif environment == "kindest" and KINDEST_AVAILABLE:
        env = Platform2DEnvKindest(render_mode=render_mode, render_every=env_render_every)
        obs_dim = OBS_DIM_SENSING  # Same as sensing
        print(f"✅ Using KINDEST environment (obs_dim={obs_dim}) - auto-eat + proximity rewards")
    else:
        env = Platform2DEnv(render_mode=render_mode, render_every=env_render_every)
        obs_dim = OBS_DIM
        print(f"✅ Using base environment (obs_dim={obs_dim})")
    
    if render_mode is None:
        print("⚡ Rendering disabled for maximum speed")
    else:
        print(f"📺 Rendering: Every step in episodes 0, {render_every}, {render_every*2}, ... (every {render_every} episodes)")
    
    # Initialize agent with correct observation dimension
    agent = AdaptivePPOAgent(
        obs_dim=obs_dim,
        action_dim=NUM_ACTIONS,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device
    )
    
    # Load existing agent if specified (transfer learning)
    start_episode = 0
    if load_agent:
        print(f"📂 Loading agent from: {load_agent}")
        metadata = agent.load(load_agent)
        if metadata:
            print(f"   Previous agent name: {metadata.get('agent_name', 'Unknown')}")
            print(f"   Previous environment: {metadata.get('environment', 'Unknown')}")
            print(f"   Previous episode: {metadata.get('episode', 'Unknown')}")
        else:
            print("   (No metadata found - agent will keep existing name)")
    
    # Initialize logger
    logger = TrainingLogger(log_dir)
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Render every: {render_every} episodes")
    print(f"Save every: {save_every} episodes")
    if load_agent:
        print(f"Transfer learning from: {load_agent}")
    print("=" * 80 + "\n")
    
    # Training loop
    for episode in range(episodes):
        # Pass episode number to environment for display
        obs, info = env.reset(options={'episode': episode})
        episode_reward = 0
        episode_length = 0
        episode_health = []
        episode_hunger = []
        episode_risk = []
        
        # No artificial step limit - let agent live as long as it survives
        step = 0
        while True:
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, value, log_prob, terminated or truncated)
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            step += 1
            episode_health.append(info['health'])
            episode_hunger.append(info['hunger'])
            episode_risk.append(info['risk'])
            
            # Render environment - only render every step in episodes that are multiples of render_every
            # Skip if headless (render_mode is None)
            if render_mode == "human" and episode % render_every == 0:
                env.render()
            
            # Check termination - episode ends when agent dies (or hits environment limit)
            if terminated or truncated:
                break
            
            obs = next_obs
        
        # Update agent if buffer is full
        if agent.buffer.size >= update_frequency:
            losses = agent.update()
            logger.log_training(losses)
        
        # Log episode
        avg_health = np.mean(episode_health) if episode_health else 0
        avg_hunger = np.mean(episode_hunger) if episode_hunger else 0
        avg_risk = np.mean(episode_risk) if episode_risk else 0
        
        # Optimize logging for headless mode
        if render_mode is None:
            # Headless: minimal logging - just track arrays, print every 10 episodes
            logger.episode_rewards.append(episode_reward)
            logger.episode_lengths.append(episode_length)
            logger.episode_health.append(avg_health)
            logger.episode_hunger.append(avg_hunger)
            logger.episode_risk.append(avg_risk)
            # Update rolling windows (deque handles maxlen automatically)
            logger.reward_window.append(episode_reward)
            logger.length_window.append(episode_length)
            logger.health_window.append(avg_health)
            logger.hunger_window.append(avg_hunger)
            logger.risk_window.append(avg_risk)
            
            # Print stats only every 10 episodes
            if episode % 10 == 0 or episode == 1:
                logger.print_stats(episode)
        else:
            # Rendering mode: full logging every episode
            logger.log_episode(episode_reward, episode_length, avg_health, avg_hunger, avg_risk)
            logger.print_stats(episode)
        # logger.update_plots(episode)  # Disabled to prevent blank windows
        
        # Print maze completion info
        if info.get('maze_completed', False):
            print(f"🎉 MAZE COMPLETED in episode {episode}!")
            print(f"   Food collected: {info['food_collected']}/{info['total_food']}")
            print(f"   Final health: {info['health']:.1f}")
            print(f"   Steps taken: {info['step']}")
            print(f"   Episode reward: {info['episode_reward']:.2f}")
        
        # Save model
        if episode % save_every == 0 and episode > 0:
            # Save with agent name and environment metadata
            model_path = os.path.join(model_dir, f"{agent_name}_episode_{episode}.pth")
            agent.save(model_path, agent_name=agent_name, environment=environment, episode=episode)
            print(f"💾 Model saved: {model_path}")
            
            # Save logs when saving model (much less frequent in headless)
            if render_mode is None:
                # Headless: only save logs every 1000 episodes with models
                if episode % 1000 == 0:
                    logger.save_logs()
            else:
                # Rendering mode: save logs with every model save
                logger.save_logs()
    
    # Final save
    final_model_path = os.path.join(model_dir, f"{agent_name}_final.pth")
    agent.save(final_model_path, agent_name=agent_name, environment=environment, episode=episodes-1)
    # Always save final logs
    logger.save_logs()
    
    # In headless mode, save logs less frequently during training
    if render_mode is None:
        print("⚡ Optimized for speed: Logs saved every 1000 episodes")
    
    print("Training completed!")
    print(f"Final model saved to {final_model_path}")
    
    # Keep plots open
    plt.ioff()
    plt.show()
    
    env.close()


def test_agent(model_path: str, episodes: int = 5, render: bool = True):
    """Test a trained agent."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize environment and agent
    env = Platform2DEnv(render_mode="human" if render else None, render_every=1)
    agent = AdaptivePPOAgent(
        obs_dim=OBS_DIM,
        action_dim=NUM_ACTIONS,
        device=device
    )
    
    # Load model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Test episodes
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\nTest Episode {episode + 1}")
        print("-" * 40)
        
        for step in range(1000):
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for viewing
            
            if terminated or truncated:
                break
        
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length}")
        print(f"Final Health: {info['health']:.1f}")
        print(f"Final Hunger: {info['hunger']:.1f}")
        print(f"Final Risk: {info['risk']:.1f}")
    
    env.close()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train or test PPO agent on 2D Platform environment")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode: train or test")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render-every", type=int, default=10, help="Render every N episodes")
    parser.add_argument("--save-every", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--model-path", type=str, help="Path to model for testing")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--update-freq", type=int, default=2048, help="Update frequency")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.mode == "train":
        train_agent(
            episodes=args.episodes,
            max_steps=args.max_steps,
            render_every=args.render_every,
            save_every=args.save_every,
            device=device,
            lr=args.lr,
            gamma=args.gamma,
            eps_clip=args.eps_clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            update_frequency=args.update_freq
        )
    elif args.mode == "test":
        if args.model_path is None:
            print("Error: --model-path is required for test mode")
            return
        test_agent(args.model_path, episodes=args.episodes, render=True)


if __name__ == "__main__":
    main()
