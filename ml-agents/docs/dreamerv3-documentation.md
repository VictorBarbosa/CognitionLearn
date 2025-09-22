# DreamerV3 (Mastering Diverse Domains through World Models)

## Overview

DreamerV3 is a state-of-the-art reinforcement learning algorithm that masters diverse domains through world models. It learns a model of the environment and uses it for planning, making it highly sample-efficient and capable of handling complex tasks.

## Key Features

1. **World Model**: Learns a complete model of the environment including dynamics, rewards, and observations
2. **Latent Space Planning**: Plans actions in a compact latent space rather than raw observation space
3. **Sample Efficiency**: Extremely efficient in terms of samples needed for learning
4. **Multi-Task Learning**: Can handle diverse domains with a single set of hyperparameters
5. **Imagination-Based Learning**: Uses imagined trajectories for policy improvement

## Configuration Parameters

### Core Parameters
- `batch_size`: Batch size for training
- `buffer_size`: Size of the replay buffer
- `buffer_init_steps`: Number of steps to initialize the buffer
- `tau`: Target network update rate
- `horizon`: Planning horizon
- `imagination_horizon`: Horizon for imagination

### DreamerV3 Specific Parameters
- `kl_scale`: Scale for KL divergence loss
- `kl_balance`: Balance between forward and backward KL
- `kl_free`: Free KL term
- `kl_forward`: Weight for forward KL
- `kl_backward`: Weight for backward KL
- `discount`: Discount factor
- `lambda_`: Lambda for GAE
- `adam_epsilon`: Epsilon for Adam optimizer
- `grad_clip`: Gradient clipping value
- `latent_size`: Size of latent representation
- `deter_size`: Size of deterministic state
- `stoch_size`: Size of stochastic state
- `hidden_size`: Size of hidden layers
- `embed_size`: Size of observation embedding
- `reward_layers`: Number of layers for reward model
- `discount_layers`: Number of layers for discount model
- `actor_layers`: Number of layers for actor
- `critic_layers`: Number of layers for critic
- `cnn_depth`: Depth for CNN layers
- `dense_layers`: Number of dense layers
- `ensemble`: Ensemble size
- `pretrain`: Pretraining steps
- `train_world_model`: Whether to train world model
- `train_actor`: Whether to train actor
- `train_critic`: Whether to train critic

## How it Works

The DreamerV3 algorithm consists of three main components:

1. **World Model**:
   - **Encoder**: Maps observations to latent representations
   - **RSSM (Recurrent State Space Model)**: Models dynamics in latent space
   - **Decoder**: Reconstructs observations from latent states
   - **Reward Model**: Predicts rewards from latent states

2. **Actor-Critic**:
   - **Actor**: Policy that operates in latent space
   - **Critic**: Value function that evaluates latent states

3. **Training Process**:
   - Learn world model from real experiences
   - Use world model to imagine future trajectories
   - Train actor-critic on imagined trajectories
   - Collect real experiences and repeat

## When to Use DreamerV3

DreamerV3 is particularly useful in scenarios where:

1. **Sample Efficiency is Critical**: When you have limited interaction with the environment
2. **Complex Environments**: Environments with high-dimensional observations or complex dynamics
3. **Multi-Task Learning**: When you need to solve multiple tasks with a single algorithm
4. **Offline Learning**: When you want to learn from previously collected data
5. **Long-Horizon Tasks**: Tasks that require planning over long time horizons

## Example Configuration

```yaml
default:
  trainer: dreamerv3
  batch_size: 50
  buffer_size: 1000000
  buffer_init_steps: 1000
  hidden_units: 256
  learning_rate: 3.0e-4
  max_steps: 1.0e6
  tau: 0.005
  # DreamerV3 specific parameters
  horizon: 15
  imagination_horizon: 15
  kl_scale: 1.0
  kl_balance: 0.8
  discount: 0.99
  latent_size: 32
  deter_size: 256
  stoch_size: 32
  hidden_size: 256
  embed_size: 256
  reward_layers: 2
  actor_layers: 2
  critic_layers: 2
  reward_signals:
    extrinsic:
      strength: 1.0
      gamma: 0.99
```

## Architecture

The DreamerV3 architecture consists of:

1. **World Model**:
   - Encoder: Maps observations to embeddings
   - RSSM: Recurrent State Space Model for dynamics
   - Decoder: Maps latent states to observations
   - Reward Model: Predicts rewards from latent states

2. **Actor-Critic**:
   - Actor: Policy network in latent space
   - Critic: Value network in latent space

3. **Training Loop**:
   - Collect real experiences
   - Train world model on real experiences
   - Use world model to imagine trajectories
   - Train actor-critic on imagined trajectories
   - Repeat

The world model enables DreamerV3 to learn efficiently by leveraging imagination for policy improvement, making it one of the most sample-efficient RL algorithms available.