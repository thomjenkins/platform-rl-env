# Environment Variants

This document describes the three different environment variants for training the RL agent.

## Environment 1: Base Environment (`env_platform.py`)

**Class:** `Platform2DEnv`

**Features:**
- ✅ Automatic food eating on proximity (no ACTION_EAT required)
- ❌ No food proximity sensing in observations
- ❌ No proximity-based reward shaping

**Observation Space:** `[x, y, vx, vy, health, hunger, risk]` (7 dimensions)

**Use Case:** Baseline environment for initial learning.

---

## Environment 2: Sensing Environment (`env_platform_sensing.py`)

**Class:** `Platform2DEnvSensing`

**Features:**
- ❌ **Requires ACTION_EAT** - agent must explicitly eat food
- ✅ Food proximity sensing in observations
- ❌ No proximity-based reward shaping

**Observation Space:** `[x, y, vx, vy, health, hunger, risk, food_dist, food_dx, food_dy]` (10 dimensions)

**Use Case:** Agent gets information about food location (making it easier to find) but must learn to use ACTION_EAT to consume it. Good middle ground between base and proximity_reward.

---

## Environment 3: Proximity Reward Environment (`env_platform_proximity_reward.py`)

**Class:** `Platform2DEnvProximityReward`

**Features:**
- ❌ **Requires ACTION_EAT** - agent must explicitly eat food
- ✅ Food proximity sensing in observations
- ✅ **Proximity-based reward shaping** (small rewards for getting closer)

**Observation Space:** `[x, y, vx, vy, health, hunger, risk, food_dist, food_dx, food_dy]` (10 dimensions)

**Reward Structure:**
- **Proximity reward:** ±1.0 max per step (scaled by 0.1 per unit distance change)
- **Food collection reward:** 100+ (base + hunger bonus + health bonus + speed bonus)
- **Design guarantee:** Food collection reward always outweighs proximity penalty

**Use Case:** Most challenging environment - requires learning ACTION_EAT while getting reward guidance toward food.

---

## Summary Table

| Environment | Auto-Eat | Sensing | Proximity Reward | ACTION_EAT Required | Observation Dims |
|------------|----------|---------|------------------|---------------------|------------------|
| Base | ✅ | ❌ | ❌ | ❌ | 7 |
| Sensing | ❌ | ✅ | ❌ | ✅ | 10 |
| Proximity Reward | ❌ | ✅ | ✅ | ✅ | 10 |

## Training Progression

Recommended curriculum:
1. **Start with Base** - Learn basic navigation and survival (auto-eat makes it easier)
2. **Move to Sensing** - Learn ACTION_EAT with food location information (easiest way to learn eating)
3. **Advance to Proximity Reward** - Add proximity reward shaping for most complete challenge

Or use any single environment for focused training on specific skills.

