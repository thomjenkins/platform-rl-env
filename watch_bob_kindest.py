"""
Watch Bob learn in the kindest environment with full rendering.
Shows every step of every episode for 1000 episodes.
"""

from train import train_agent
import os

if __name__ == "__main__":
    print("=" * 80)
    print("👀 Watching Bob Learn in Kindest Environment")
    print("=" * 80)
    
    # Load Bob's latest checkpoint for transfer learning
    bob_dir = "models/Bob"
    load_agent = None
    
    if os.path.exists(bob_dir):
        final_model = os.path.join(bob_dir, "Bob_final.pth")
        if os.path.exists(final_model):
            load_agent = final_model
        else:
            # Find latest episode checkpoint
            checkpoint_files = [f for f in os.listdir(bob_dir) if f.endswith(".pth") and "episode_" in f]
            if checkpoint_files:
                episodes = []
                for f in checkpoint_files:
                    try:
                        ep_num = int(f.split("episode_")[1].split(".pth")[0])
                        episodes.append((ep_num, f))
                    except:
                        pass
                if episodes:
                    latest_ep, latest_file = max(episodes, key=lambda x: x[0])
                    load_agent = os.path.join(bob_dir, latest_file)
    
    if load_agent:
        print(f"📂 Curriculum Learning: Transferring from base environment")
        print(f"   Loading: {os.path.basename(load_agent)}")
        print(f"   Agent will adapt input layer for new observations (food proximity sensing)")
    else:
        print("🚀 Starting fresh (no previous checkpoint found)")
    
    print("\n" + "=" * 80)
    print("🎬 FULL RENDERING MODE - Watching every step of every episode!")
    print("=" * 80)
    
    train_agent(
        episodes=1000,  # 1000 episodes
        agent_name="Bob",
        environment="kindest",
        model_dir="models/Bob",
        log_dir="logs/Bob",
        render_every=1,  # Render every episode (every step)
        save_every=100,  # Save every 100 episodes
        max_steps=1000,
        load_agent=load_agent  # Transfer learning enabled
    )
    
    print("\n✅ Training complete!")

