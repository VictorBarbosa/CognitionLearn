# PPO-ET (Proximal Policy Optimization with Entropy Temperature)

## Overview

PPO-ET is an extension of the standard Proximal Policy Optimization (PPO) algorithm that incorporates adaptive entropy temperature control. This enhancement allows for better exploration-exploitation trade-off during training by dynamically adjusting the entropy coefficient based on the policy's entropy.

## Key Features

1. **Entropy Temperature Control**: Dynamically adjusts the entropy coefficient during training
2. **Adaptive Exploration**: Automatically balances exploration and exploitation
3. **Target Entropy**: Can specify a target entropy value or let the algorithm calculate it automatically
4. **Compatibility**: Maintains full compatibility with existing ML-Agents features

## Configuration Parameters

### Core PPO Parameters
- `beta`: Entropy coefficient (initial value when using adaptive control)
- `epsilon`: Clipping parameter for PPO
- `lambd`: Lambda parameter for GAE
- `num_epoch`: Number of epochs for learning

### PPO-ET Specific Parameters
- `entropy_temperature`: Initial entropy temperature value
- `adaptive_entropy_temperature`: Whether to use adaptive entropy temperature control
- `target_entropy`: Target entropy value (if null, automatically calculated)

## How it Works

The PPO-ET algorithm extends the standard PPO loss function by incorporating an adaptive entropy term:

```
Loss = L_policy + 0.5 * L_value - α * H(π)
```

Where:
- `L_policy` is the policy loss
- `L_value` is the value loss
- `α` is the entropy temperature coefficient
- `H(π)` is the policy entropy

When `adaptive_entropy_temperature` is enabled, the algorithm learns the optimal entropy temperature `α` by minimizing the difference between the current policy entropy and the target entropy.

## When to Use PPO-ET

PPO-ET is particularly useful in scenarios where:

1. **Complex Environments**: Environments with sparse rewards or complex exploration requirements
2. **Fine-tuning**: When you need more control over the exploration-exploitation trade-off
3. **Stable Training**: When you want the stability of PPO with improved exploration capabilities

## Example Configuration

```yaml
default:
  trainer: ppo_et
  batch_size: 1024
  beta: 5.0e-3
  buffer_size: 10240
  epsilon: 0.2
  hidden_units: 128
  lambd: 0.95
  learning_rate: 3.0e-4
  max_steps: 5.0e5
  num_epoch: 3
  entropy_temperature: 1.0
  adaptive_entropy_temperature: true
  reward_signals:
    extrinsic:
      strength: 1.0
      gamma: 0.99
```