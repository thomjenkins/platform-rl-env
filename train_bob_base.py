"""
Train Bob from scratch (no prior learning) in the base environment for 1000 episodes.
"""

from train import train_agent

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 Training Bob in Base Environment (from scratch)")
    print("=" * 80)
    
    train_agent(
        episodes=1000,
        agent_name="Bob",
        environment="base",
        model_dir="models/Bob",
        log_dir="logs/Bob",
        render_every=50,  # Render every N episodes (change this number to adjust frequency)
        save_every=100,
        max_steps=1000,
        load_agent=None  # No prior learning - fresh start
    )

