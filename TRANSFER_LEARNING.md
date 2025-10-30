# Transfer Learning & Curriculum Learning

## Overview

The agent now supports **transfer learning** between environments with different observation dimensions. This enables **curriculum learning**, where an agent can:

1. Learn fundamental skills in a simpler environment (e.g., base environment)
2. Transfer those learned skills to a more complex environment (e.g., kindest environment)
3. Adapt and learn new capabilities specific to the new environment

## How It Works

When loading a checkpoint from an environment with a different `obs_dim`:

1. **Compatible Layers**: All hidden layers and output layers are loaded (they share the same architecture)
2. **Input Layer**: The input layer is kept randomly initialized to learn new observation features
3. **Knowledge Transfer**: The agent retains learned representations and strategies while adapting to new sensory inputs

### Example Curriculum Path

```
Bob's Learning Journey:
  Step 1: Base Environment (obs_dim=7)
    - Learn basic movement, survival, food finding
    - Trains for ~1000 episodes
    - Saves: models/Bob/Bob_episode_1000.pth

  Step 2: Kindest Environment (obs_dim=10) 
    - Transfers learned behaviors from base
    - Input layer learns to interpret food proximity sensing
    - Leverages existing movement/survival strategies
    - Trains for ~100,000 episodes in enhanced environment
```

## Technical Details

- **Transfer Learning Flag**: `agent.load(filepath, transfer_learning=True)` (default: True)
- **Layer Matching**: Automatically detects shape mismatches and handles them gracefully
- **Optimizer Reset**: When input dimensions change, optimizers start fresh (they adapt quickly)

## Benefits

1. **Faster Learning**: Agent doesn't start from scratch in new environments
2. **Stable Training**: Learned representations provide stable foundation
3. **Natural Progression**: Curriculum from simple to complex tasks
4. **Knowledge Retention**: Core strategies (movement, risk assessment) transfer across environments

