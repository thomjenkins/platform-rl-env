"""
Train Bob for 1000 episodes in the sensing environment.
"""

from train import train_agent

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 Training Bob in Sensing Environment")
    print("=" * 80)
    
    train_agent(
        episodes=1000,
        agent_name="Bob",
        environment="sensing",
        model_dir="models/Bob",
        log_dir="logs/Bob",
        render_every=10,
        save_every=100,
        max_steps=1000
    )

