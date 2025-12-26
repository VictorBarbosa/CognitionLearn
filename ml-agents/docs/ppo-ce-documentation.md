# PPO-CE (Proximal Policy Optimization with Curiosity Exploration)

## Overview

PPO-CE is an extension of the standard Proximal Policy Optimization (PPO) algorithm that incorporates curiosity-driven exploration. This enhancement allows for better exploration in environments with sparse rewards by providing an intrinsic reward based on the agent's curiosity.

## Key Features

1. **Curiosity-Driven Exploration**: Adds intrinsic rewards based on the agent's curiosity
2. **Transition Model**: Learns to predict the next state given the current state and action
3. **Imagination-Augmented**: Can use imagined trajectories to improve learning
4. **Compatibility**: Maintains full compatibility with existing ML-Agents features

## Configuration Parameters

### Core PPO Parameters
- `beta`: Entropy coefficient
- `epsilon`: Clipping parameter for PPO
- `lambd`: Lambda parameter for GAE
- `num_epoch`: Number of epochs for learning

### PPO-CE Specific Parameters
- `curiosity_strength`: Strength of the curiosity reward
- `curiosity_gamma`: Discount factor for curiosity rewards
- `curiosity_learning_rate`: Learning rate for the curiosity module
- `curiosity_hidden_units`: Number of hidden units in curiosity networks
- `curiosity_num_layers`: Number of layers in curiosity networks
- `imagination_horizon`: Horizon for imagination-augmented planning
- `use_imagination_augmented`: Whether to use imagination-augmented planning
- `curiosity_loss_weight`: Weight for the curiosity loss

## How it Works

The PPO-CE algorithm extends the standard PPO by adding a curiosity module that computes intrinsic rewards:

```
r_total = r_extrinsic + β * r_curiosity
```

Where:
- `r_extrinsic` is the environment reward
- `β` is the curiosity strength
- `r_curiosity` is the curiosity reward based on prediction error

The curiosity module consists of:
1. **State Encoder**: Encodes observations into latent representations
2. **Action Encoder**: Encodes actions into latent representations
3. **Transition Network**: Predicts the next state encoding given current state and action
4. **Curiosity Reward**: Computed as the error between predicted and actual next state

## When to Use PPO-CE

PPO-CE is particularly useful in scenarios where:

1. **Sparse Reward Environments**: Environments with infrequent or rare rewards
2. **Exploration-Intensive Tasks**: Tasks that require significant exploration to solve
3. **Complex State Spaces**: Environments with high-dimensional or complex observations
4. **Transfer Learning**: When pre-training in a source environment to improve performance in a target environment

## Example Configuration

```yaml
default:
  trainer: ppo_ce
  batch_size: 1024
  beta: 5.0e-3
  buffer_size: 10240
  epsilon: 0.2
  hidden_units: 128
  lambd: 0.95
  learning_rate: 3.0e-4
  max_steps: 5.0e5
  num_epoch: 3
  curiosity_strength: 0.01
  curiosity_gamma: 0.99
  curiosity_learning_rate: 3e-4
  curiosity_hidden_units: 128
  curiosity_num_layers: 2
  reward_signals:
    extrinsic:
      strength: 1.0
      gamma: 0.99
```

## Architecture

The PPO-CE architecture consists of:

1. **Policy Network**: Standard PPO policy network
2. **Value Network**: Standard PPO value network
3. **Curiosity Module**:
   - State Encoder: Encodes observations
   - Action Encoder: Encodes actions
   - Transition Network: Predicts next state
   - Curiosity Reward Calculator: Computes intrinsic rewards

The curiosity module is trained to minimize the prediction error between actual and predicted next states, which encourages the agent to explore novel states that are difficult to predict.