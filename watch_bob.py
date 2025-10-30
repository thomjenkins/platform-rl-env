"""
Watch Bob perform in the environment with full rendering enabled.

Loads a trained Bob checkpoint and runs episodes with visualization.
"""

import argparse
import os
from env_platform import Platform2DEnv
from agent import AdaptivePPOAgent
from utils import OBS_DIM, NUM_ACTIONS

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


def watch_agent(agent_path: str, environment: str = "base", episodes: int = 5):
    """
    Watch an agent perform with full rendering.
    
    Args:
        agent_path: Path to agent checkpoint (.pth file)
        environment: Environment to watch in ("base", "sensing", "proximity_reward")
        episodes: Number of episodes to watch
    """
    print("=" * 80)
    print(f"👀 Watching Agent: {os.path.basename(agent_path)}")
    print(f"🌍 Environment: {environment}")
    print(f"📺 Episodes: {episodes}")
    print("=" * 80)
    
    # Initialize environment with rendering enabled
    if environment == "sensing" and SENSING_AVAILABLE:
        env = Platform2DEnvSensing(render_mode="human", render_every=1)
        obs_dim = OBS_DIM_SENSING
    elif environment == "proximity_reward" and PROXIMITY_REWARD_AVAILABLE:
        env = Platform2DEnvProximityReward(render_mode="human", render_every=1)
        obs_dim = OBS_DIM_SENSING
    else:
        env = Platform2DEnv(render_mode="human", render_every=1)
        obs_dim = OBS_DIM
    
    # Initialize agent
    agent = AdaptivePPOAgent(
        obs_dim=obs_dim,
        action_dim=NUM_ACTIONS,
        device="cpu"
    )
    
    # Load agent
    print(f"\n📂 Loading agent from: {agent_path}")
    metadata = agent.load(agent_path)
    if metadata:
        print(f"   Agent name: {metadata.get('agent_name', 'Unknown')}")
        print(f"   Trained in: {metadata.get('environment', 'Unknown')}")
        print(f"   Episode: {metadata.get('episode', 'Unknown')}")
    
    print("\n🎬 Starting visualization (close window to stop)...\n")
    
    # Run episodes with full rendering
    for episode in range(episodes):
        obs, info = env.reset(options={'episode': episode})
        episode_reward = 0
        step = 0
        
        print(f"Episode {episode + 1}/{episodes}")
        
        while True:
            # Get action (deterministic for best performance)
            action, _, _ = agent.get_action(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Render every step
            env.render()
            
            if terminated or truncated:
                break
        
        print(f"   Episode reward: {episode_reward:.2f}")
        print(f"   Steps: {step}")
        print(f"   Final health: {info['health']:.1f}")
        print(f"   Food collected: {info.get('food_collected', 0)}\n")
    
    env.close()
    print("Done watching!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained agent perform")
    parser.add_argument("--agent", type=str, required=True,
                       help="Path to agent checkpoint (e.g., models/Bob/Bob_final.pth)")
    parser.add_argument("--env", type=str, default="base",
                       choices=["base", "sensing", "proximity_reward"],
                       help="Environment to watch in")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to watch")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.agent):
        print(f"❌ Error: Agent file not found: {args.agent}")
        print("\nAvailable Bob checkpoints:")
        bob_dir = "models/Bob"
        if os.path.exists(bob_dir):
            for f in sorted(os.listdir(bob_dir)):
                if f.endswith(".pth"):
                    print(f"  - {os.path.join(bob_dir, f)}")
        exit(1)
    
    watch_agent(args.agent, args.env, args.episodes)

