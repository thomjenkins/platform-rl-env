"""
Training script for named agents with distinct learning histories.

Example usage:
    # Train Bob from scratch in base environment
    python train_named_agent.py --agent Bob --env base
    
    # Continue training Bob in sensing environment
    python train_named_agent.py --agent Bob --env sensing --load models/Bob_final.pth
    
    # Train Berta in proximity_reward environment
    python train_named_agent.py --agent Berta --env proximity_reward
"""

import argparse
import sys
import os

# Import appropriate environment based on argument
def get_environment_class(env_name: str):
    """Get the appropriate environment class."""
    if env_name == "base":
        from env_platform import Platform2DEnv
        from utils import OBS_DIM
        return Platform2DEnv, OBS_DIM
    elif env_name == "sensing":
        from env_platform_sensing import Platform2DEnvSensing, OBS_DIM_SENSING
        return Platform2DEnvSensing, OBS_DIM_SENSING
    elif env_name == "proximity_reward":
        from env_platform_proximity_reward import Platform2DEnvProximityReward, OBS_DIM_SENSING
        return Platform2DEnvProximityReward, OBS_DIM_SENSING
    else:
        raise ValueError(f"Unknown environment: {env_name}. Choose: base, sensing, proximity_reward")

def main():
    parser = argparse.ArgumentParser(description="Train named agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent name (e.g., Bob, Berta)")
    parser.add_argument("--env", type=str, default="base", 
                       choices=["base", "sensing", "proximity_reward"],
                       help="Environment to train in")
    parser.add_argument("--load", type=str, default=None, 
                       help="Path to previous agent checkpoint (for transfer learning)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--save-every", type=int, default=100, help="Save every N episodes")
    parser.add_argument("--render-every", type=int, default=10, help="Render every N episodes")
    
    args = parser.parse_args()
    
    # Get environment and observation dimension
    EnvClass, obs_dim = get_environment_class(args.env)
    
    # Set up directories based on agent name
    model_dir = os.path.join("models", args.agent)
    log_dir = os.path.join("logs", args.agent)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"🤖 Training Agent: {args.agent}")
    print(f"🌍 Environment: {args.env}")
    print(f"📊 Observation dim: {obs_dim}")
    if args.load:
        print(f"📂 Loading from: {args.load}")
    print("=" * 80)
    
    # Import training function and update it for named agents
    from train import train_agent
    from agent import AdaptivePPOAgent
    from utils import NUM_ACTIONS
    
    # Create environment
    env = EnvClass(render_mode="human", render_every=args.render_every)
    
    # Create agent
    agent = AdaptivePPOAgent(
        obs_dim=obs_dim,
        action_dim=NUM_ACTIONS,
        device="cpu"
    )
    
    # Load if specified
    if args.load:
        metadata = agent.load(args.load)
        if metadata:
            print(f"   Loaded agent: {metadata.get('agent_name', 'Unknown')}")
            print(f"   From environment: {metadata.get('environment', 'Unknown')}")
            print(f"   Episode: {metadata.get('episode', 'Unknown')}")
    
    # Import training loop (simplified version)
    # You can adapt the training loop from train.py here
    print(f"\nStarting training for {args.agent}...")
    print("(Full training loop would go here)")
    print("\n💡 To implement full training, see train.py for the complete loop")
    print("   For now, you can use train.py with agent_name, environment, and load_agent parameters")

if __name__ == "__main__":
    main()

