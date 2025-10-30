"""
Retrospectively render Bob performing in specific episodes.
Loads Bob's checkpoint and renders episodes 0, 1000, 2000, 3000, etc.
"""

import argparse
import os
import torch
from env_platform_kindest import Platform2DEnvKindest
from agent import AdaptivePPOAgent
from utils import NUM_ACTIONS, OBS_DIM_SENSING

def render_episodes(agent_path: str, episodes_to_render: list, environment: str = "kindest"):
    """
    Render specific episodes from a trained agent.
    
    Args:
        agent_path: Path to agent checkpoint
        episodes_to_render: List of episode numbers to render (e.g., [0, 1000, 2000, ...])
        environment: Environment to use
    """
    print("=" * 80)
    print(f"📺 Retrospective Rendering: Episodes {episodes_to_render}")
    print(f"📂 Loading agent: {os.path.basename(agent_path)}")
    print("=" * 80)
    
    # Initialize environment with full rendering
    if environment == "kindest":
        from env_platform_kindest import Platform2DEnvKindest
        env = Platform2DEnvKindest(render_mode="human", render_every=1)
        obs_dim = OBS_DIM_SENSING
    elif environment == "base":
        from env_platform import Platform2DEnv
        from utils import OBS_DIM
        env = Platform2DEnv(render_mode="human", render_every=1)
        obs_dim = OBS_DIM
    elif environment == "sensing":
        from env_platform_sensing import Platform2DEnvSensing, OBS_DIM_SENSING
        env = Platform2DEnvSensing(render_mode="human", render_every=1)
        obs_dim = OBS_DIM_SENSING
    else:
        raise ValueError(f"Unknown environment: {environment}")
    
    # Initialize and load agent
    agent = AdaptivePPOAgent(obs_dim=obs_dim, action_dim=NUM_ACTIONS, device="cpu")
    metadata = agent.load(agent_path)
    
    if metadata:
        print(f"   Agent: {metadata.get('agent_name', 'Unknown')}")
        print(f"   Trained in: {metadata.get('environment', 'Unknown')}")
        print(f"   Episode: {metadata.get('episode', 'Unknown')}")
    
    print("\n🎬 Rendering episodes (close window after each episode to continue)...\n")
    
    for ep_num in episodes_to_render:
        print(f"{'='*80}")
        print(f"Episode {ep_num}")
        print(f"{'='*80}")
        
        # Set episode number for display
        obs, info = env.reset(options={'episode': ep_num})
        episode_reward = 0
        step = 0
        
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
        
        print(f"\n   Episode {ep_num} complete:")
        print(f"   Reward: {episode_reward:.2f}")
        print(f"   Steps: {step}")
        print(f"   Final health: {info['health']:.1f}")
        print(f"   Food collected: {info.get('food_collected', 0)}/{info.get('total_food', 0)}")
        print()
        
        # Wait for user input before next episode
        input("Press Enter to continue to next episode (or Ctrl+C to stop)...")
    
    env.close()
    print("\n✅ Done rendering all episodes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrospectively render specific episodes")
    parser.add_argument("--agent", type=str, default=None,
                       help="Path to agent checkpoint (auto-detects if not provided)")
    parser.add_argument("--env", type=str, default="kindest",
                       choices=["kindest", "base", "sensing"],
                       help="Environment to render in")
    parser.add_argument("--every", type=int, default=1000,
                       help="Render every Nth episode (e.g., 1000 = episodes 0, 1000, 2000, ...)")
    parser.add_argument("--max-ep", type=int, default=100000,
                       help="Maximum episode to render up to")
    parser.add_argument("--episodes", type=str, default=None,
                       help="Comma-separated list of specific episodes (e.g., '0,1000,2000')")
    
    args = parser.parse_args()
    
    # Auto-detect agent if not provided
    if args.agent is None:
        bob_dir = "models/Bob"
        final_model = os.path.join(bob_dir, "Bob_final.pth")
        if os.path.exists(final_model):
            args.agent = final_model
        else:
            print(f"❌ Error: No agent checkpoint found. Please specify --agent")
            print(f"   Expected: {final_model}")
            exit(1)
    
    if not os.path.exists(args.agent):
        print(f"❌ Error: Agent file not found: {args.agent}")
        exit(1)
    
    # Determine which episodes to render
    if args.episodes:
        # User specified exact episodes
        episodes = [int(x.strip()) for x in args.episodes.split(",")]
    else:
        # Render every Nth episode
        episodes = list(range(0, args.max_ep + 1, args.every))
    
    print(f"\n📋 Will render {len(episodes)} episodes: {episodes[:10]}{'...' if len(episodes) > 10 else ''}\n")
    
    render_episodes(args.agent, episodes, args.env)

