# Agent Naming and Learning History System

This system allows you to create named agents (Bob, Berta, etc.) with distinct learning histories that can be transferred between environments.

## Features

✅ **Named Agents**: Give agents human-readable names (Bob, Berta, etc.)
✅ **Metadata Tracking**: Each saved model includes:
   - Agent name
   - Environment it was trained in
   - Episode number
✅ **Transfer Learning**: Load Bob trained in one environment into another environment
✅ **Separate Histories**: Each agent maintains its own learning history

## Usage Examples

### 1. Train Bob from scratch in base environment

```python
from train import train_agent

train_agent(
    episodes=1000,
    agent_name="Bob",
    environment="base",
    model_dir="models/Bob"
)
```

### 2. Transfer Bob to sensing environment

```python
train_agent(
    episodes=1000,
    agent_name="Bob",
    environment="sensing",
    load_agent="models/Bob/Bob_final.pth",
    model_dir="models/Bob"
)
```

### 3. Train new agent Berta in proximity_reward environment

```python
train_agent(
    episodes=1000,
    agent_name="Berta",
    environment="proximity_reward",
    model_dir="models/Berta"
)
```

## Directory Structure

```
models/
├── Bob/
│   ├── Bob_episode_100.pth
│   ├── Bob_episode_200.pth
│   └── Bob_final.pth
├── Berta/
│   ├── Berta_episode_100.pth
│   └── Berta_final.pth
└── ...
```

## Loading Agents

```python
from agent import AdaptivePPOAgent
from utils import OBS_DIM, NUM_ACTIONS

# Load Bob
agent = AdaptivePPOAgent(obs_dim=OBS_DIM, action_dim=NUM_ACTIONS)
metadata = agent.load("models/Bob/Bob_final.pth")

print(f"Agent: {metadata['agent_name']}")
print(f"Trained in: {metadata['environment']}")
print(f"Episodes: {metadata['episode']}")
```

## Environment Identifiers

- `"base"` - Base environment (env_platform.py)
- `"sensing"` - Sensing environment (env_platform_sensing.py)
- `"proximity_reward"` - Proximity reward environment (env_platform_proximity_reward.py)

## Listing All Agents

```bash
python list_agents.py
```

This shows:
- All agents with their names
- Which environments they trained in
- Episode numbers
- File sizes

## Transfer Learning Workflow

1. **Train Bob in base environment** (learns basic navigation)
2. **Transfer Bob to sensing environment** (adds food sensing info)
3. **Transfer Bob to proximity_reward environment** (learns ACTION_EAT + proximity rewards)

Each transfer preserves Bob's learned weights while adapting to the new environment's observation space (if different).

## Important Notes

⚠️ **Observation Space Compatibility**: 
- Base: 7 dims
- Sensing: 10 dims (7 + food sensing)
- Proximity Reward: 10 dims (7 + food sensing)

When transferring between base and sensing/proximity_reward, you need to recreate the agent with the correct observation dimension, but the weights will still load (some layers may be incompatible).

For smooth transfer:
- Train in base → Transfer to sensing (network can adapt)
- Train in sensing → Transfer to proximity_reward (same obs dim, perfect transfer)

