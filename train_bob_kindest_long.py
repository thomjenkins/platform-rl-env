"""
Train Bob for 100,000 episodes in the kindest environment (headless for speed).
This is the most learning-friendly environment: auto-eat + sensing + proximity rewards.
"""

from train import train_agent

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 Training Bob for 100,000 Episodes in Kindest Environment")
    print("=" * 80)
    
    # Load Bob's latest checkpoint for transfer learning
    # This implements curriculum learning: transfer knowledge from base env to kindest env
    import os
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
    
    train_agent(
        episodes=100000,  # 100,000 episodes
        agent_name="Bob",  # Same name - continuing Bob's learning journey
        environment="kindest",
        model_dir="models/Bob",
        log_dir="logs/Bob",
        render_every=101,  # Disable rendering (headless) - >100 triggers headless mode
        save_every=1000,  # Save every 1000 episodes
        max_steps=1000,
        load_agent=load_agent  # Transfer learning enabled
    )
    
    print("\n✅ Training complete! Use render_bob_retrospective.py to view episodes.")

