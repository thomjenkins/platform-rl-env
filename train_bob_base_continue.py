"""
Continue training Bob for another 1000 episodes in the base environment.
Loads Bob's latest checkpoint and continues training.
"""

import os
from train import train_agent

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 Continuing Bob's Training in Base Environment")
    print("=" * 80)
    
    # Find Bob's latest checkpoint
    bob_dir = "models/Bob"
    if not os.path.exists(bob_dir):
        print(f"❌ Error: Bob's model directory not found: {bob_dir}")
        exit(1)
    
    # Look for final model first, then latest episode checkpoint
    final_model = os.path.join(bob_dir, "Bob_final.pth")
    if os.path.exists(final_model):
        load_agent = final_model
        print(f"📂 Loading: {load_agent}")
    else:
        # Find latest episode checkpoint
        checkpoint_files = [f for f in os.listdir(bob_dir) if f.endswith(".pth")]
        if not checkpoint_files:
            print(f"❌ Error: No checkpoints found in {bob_dir}")
            exit(1)
        
        # Extract episode numbers and get the latest
        episodes = []
        for f in checkpoint_files:
            if "episode_" in f:
                try:
                    ep_num = int(f.split("episode_")[1].split(".pth")[0])
                    episodes.append((ep_num, f))
                except:
                    pass
        
        if episodes:
            latest_ep, latest_file = max(episodes, key=lambda x: x[0])
            load_agent = os.path.join(bob_dir, latest_file)
            print(f"📂 Loading: {load_agent} (episode {latest_ep})")
        else:
            # Fallback to any .pth file
            load_agent = os.path.join(bob_dir, checkpoint_files[0])
            print(f"📂 Loading: {load_agent}")
    
    train_agent(
        episodes=1000,  # Another 1000 episodes
        agent_name="Bob",
        environment="base",
        model_dir="models/Bob",
        log_dir="logs/Bob",
        render_every=50,  # Render every N episodes (change this number to adjust frequency)
        save_every=100,
        max_steps=1000,
        load_agent=load_agent  # Continue from checkpoint
    )

