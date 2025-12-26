# CognitionLearn - Advanced ML-Agents Interface

CognitionLearn is an enhanced version of the original Unity ML-Agents Toolkit with a modern GUI interface. The `cognition-learn` package provides a set of reinforcement and imitation learning algorithms designed to be used with Unity environments. The algorithms interface with the Python API provided by the `mlagents_envs` package.

The algorithms can be accessed using the: `cognition-learn` access point with a modern GUI interface. See [here](../com.unity.ml-agents/Documentation~/Training-ML-Agents.md) for more information on using this package.

## Features

- Modern GUI interface with dark mode support
- Easy configuration of training parameters
- Real-time training control and monitoring
- Intuitive user experience for ML-Agents training

## Installation

Install the `mlagents` package (CognitionLearn is built on top of it) with:

```sh
python -m pip install mlagents==1.1.0
```

## Usage

To launch the CognitionLearn GUI interface, simply run:

```sh
cognition-learn
```

## Usage & More Information

For more information on the ML-Agents Toolkit and how to instrument a Unity
scene with the ML-Agents SDK, check out the main
[ML-Agents Toolkit documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest).

## Limitations

- Resuming self-play from a checkpoint resets the reported ELO to the default
  value.
