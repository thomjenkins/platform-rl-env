# Platform RL Environment

A 2D platform environment for reinforcement learning experiments with PPO (Proximal Policy Optimization) agents. This project features multiple environment variants, agent management, and transfer learning capabilities.

## 🎮 Environment Features

- **2D Platform Physics**: Realistic gravity, jumping, and movement mechanics
- **Dynamic Risk System**: Agent risk increases with low health and high hunger
- **Adaptive Discount Factor**: Planning horizon adjusts based on agent state
- **Food Collection**: Multiple food items with proximity sensing and reward shaping
- **Enemy AI**: Moving adversaries that damage the agent
- **Safe Zones**: Healing areas that provide health restoration

## 🌍 Environment Variants

1. **Base Environment** (`env_platform.py`): Standard platform environment
2. **Sensing Environment** (`env_platform_sensing.py`): Adds food proximity sensing
3. **Proximity Reward Environment** (`env_platform_proximity_reward.py`): Includes proximity-based reward shaping
4. **Kindest Environment** (`env_platform_kindest.py`): Combines auto-eat with proximity rewards

## 🤖 Agent Features

- **PPO Algorithm**: Proximal Policy Optimization for stable learning
- **Transfer Learning**: Agents can transfer knowledge between environments
- **Named Agents**: Track and manage multiple agents with different learning histories
- **Adaptive Behavior**: Risk-based exploration and decision making

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/platform-rl-env.git
   cd platform-rl-env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Train a New Agent
```bash
python train.py --agent_name "Alice" --environment "kindest" --episodes 1000
```

### Watch a Trained Agent
```bash
python watch_agent.py --agent_path "models/Alice/Alice_final.pth" --environment "kindest" --episodes 10
```

### List Available Agents
```bash
python list_agents.py
```

## 📊 Training Scripts

- `train_bob_base.py`: Train Bob in base environment
- `train_bob_kindest_100.py`: Train Bob for 100 episodes in kindest environment
- `train_bob_kindest_1000.py`: Train Bob for 1000 episodes in kindest environment
- `watch_bob.py`: Watch Bob perform with full rendering

## 🎯 Key Components

### Environment Files
- `env_platform.py`: Base 2D platform environment
- `env_platform_sensing.py`: Environment with food proximity sensing
- `env_platform_proximity_reward.py`: Environment with proximity reward shaping
- `env_platform_kindest.py`: Most learning-friendly environment

### Agent Files
- `agent.py`: PPO agent implementation with transfer learning
- `train.py`: Main training script with agent management

### Utility Files
- `utils.py`: Utility functions and constants
- `visual_effects.py`: Enhanced HUD and rendering effects

## 🔧 Configuration

### Environment Parameters
- `WORLD_WIDTH`: Width of the game world (800px)
- `WORLD_HEIGHT`: Height of the game world (600px)
- `MAX_HEALTH`: Maximum agent health (100)
- `MAX_HUNGER`: Maximum agent hunger (100)
- `MAX_RISK`: Maximum agent risk level (100)

### Training Parameters
- `learning_rate`: PPO learning rate (3e-4)
- `gamma`: Discount factor (adaptive)
- `eps_clip`: PPO clipping parameter (0.2)
- `k_epochs`: Number of epochs per update (4)

## 📈 Monitoring

The training process includes comprehensive logging:
- Episode rewards, lengths, and statistics
- Health, hunger, and risk tracking
- Policy and value network losses
- Food collection metrics
- Proximity reward analysis

Logs are saved to `logs/[AgentName]/` directory in JSON format.

## 🎨 Visualization

The environment includes rich visualizations:
- Real-time agent movement and actions
- Health, hunger, and risk meters
- Food proximity indicators
- Action history display
- Performance graphs and statistics

## 🔄 Transfer Learning

Agents can transfer knowledge between environments:
- Preserve learned policies from compatible layers
- Adapt input layers for new observation spaces
- Maintain learning history across environments

## 📝 Examples

### Basic Training
```python
from train import train_agent

# Train a new agent
train_agent(
    agent_name="Alice",
    environment="kindest",
    episodes=1000,
    render_every=50
)
```

### Transfer Learning
```python
# Load existing agent and transfer to new environment
train_agent(
    agent_name="Alice",
    environment="proximity_reward",
    episodes=500,
    load_agent="models/Alice/Alice_final.pth"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch for the deep learning framework
- Pygame for the 2D graphics and physics
- OpenAI Gym for environment interface inspiration
- Stable Baselines3 for PPO implementation reference

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub or contact the maintainers.

---

**Happy Learning! 🎓**