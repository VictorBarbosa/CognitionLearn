# SAC-AE (Soft Actor-Critic with Auto-Encoder)

## Overview

SAC-AE is an extension of the Soft Actor-Critic (SAC) algorithm that incorporates auto-encoder and world model components for improved learning in environments with high-dimensional observations, particularly visual inputs. This approach learns compact latent representations of observations and uses them for more efficient policy learning.

## Key Features

1. **Auto-Encoder for Representation Learning**: Learns compact latent representations of high-dimensional observations
2. **World Model**: Predicts transitions and rewards in the latent space
3. **Efficient Learning**: Operates in latent space rather than raw observation space
4. **Compatibility**: Maintains full compatibility with existing ML-Agents features

## Configuration Parameters

### Core SAC Parameters
- `batch_size`: Batch size for training
- `buffer_size`: Size of the replay buffer
- `buffer_init_steps`: Number of steps to initialize the buffer
- `tau`: Target network update rate
- `init_entcoef`: Initial entropy coefficient

### SAC-AE Specific Parameters
- `latent_size`: Size of the latent representation
- `ae_learning_rate`: Learning rate for the auto-encoder
- `ae_hidden_units`: Number of hidden units in auto-encoder networks
- `ae_num_layers`: Number of layers in auto-encoder networks
- `world_model_learning_rate`: Learning rate for the world model
- `world_model_hidden_units`: Number of hidden units in world model networks
- `world_model_num_layers`: Number of layers in world model networks
- `use_autoencoder`: Whether to use the auto-encoder
- `use_world_model`: Whether to use the world model
- `ae_loss_weight`: Weight for the auto-encoder loss
- `world_model_loss_weight`: Weight for the world model loss
- `reconstruction_loss_weight`: Weight for the reconstruction loss

## How it Works

The SAC-AE algorithm extends the standard SAC by adding representation learning components:

1. **Auto-Encoder**:
   - Encoder: Maps high-dimensional observations to compact latent representations
   - Decoder: Reconstructs observations from latent representations
   - Trained to minimize reconstruction error

2. **World Model**:
   - Transition Model: Predicts next latent state given current latent state and action
   - Reward Model: Predicts reward given latent state and action

3. **SAC in Latent Space**:
   - Policy and Q-functions operate on latent representations rather than raw observations
   - More efficient learning due to reduced dimensionality

## When to Use SAC-AE

SAC-AE is particularly useful in scenarios where:

1. **High-Dimensional Observations**: Environments with visual inputs or large observation spaces
2. **Sample Efficiency**: When you need to learn efficiently from limited data
3. **Transfer Learning**: When pre-training representations can improve performance
4. **Complex Environments**: Environments where raw observations contain redundant information

## Example Configuration

```yaml
default:
  trainer: sac_ae
  batch_size: 128
  buffer_size: 50000
  buffer_init_steps: 1000
  hidden_units: 256
  init_entcoef: 0.1
  learning_rate: 3.0e-4
  max_steps: 1.0e6
  tau: 0.005
  # SAC-AE specific parameters
  latent_size: 512
  ae_learning_rate: 1e-3
  ae_hidden_units: 256
  ae_num_layers: 2
  world_model_learning_rate: 3e-4
  world_model_hidden_units: 256
  world_model_num_layers: 2
  use_autoencoder: true
  use_world_model: true
  reward_signals:
    extrinsic:
      strength: 1.0
      gamma: 0.99
```

## Architecture

The SAC-AE architecture consists of:

1. **Policy Network**: SAC policy network operating in latent space
2. **Q-Networks**: SAC Q-networks operating in latent space
3. **Auto-Encoder**:
   - Encoder: Maps observations to latent space
   - Decoder: Maps latent space to observations
4. **World Model**:
   - Transition Model: Predicts next latent state
   - Reward Model: Predicts reward

The auto-encoder is trained to minimize reconstruction error, while the world model is trained to minimize prediction errors for transitions and rewards. The SAC components are trained in the standard way but operate on the learned latent representations.