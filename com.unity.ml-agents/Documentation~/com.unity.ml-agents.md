# ML-Agents Overview
ML-agents enable games and simulations to serve as environments for training intelligent agents in Unity. Training can be done with reinforcement learning, imitation learning, neuroevolution, or any other methods. Trained agents can be used for many use cases, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

The _ML-Agents_ package has a C# SDK for the [Unity ML-Agents Toolkit], which can be used outside of Unity. The scope of these docs is just to get started in the context of Unity, but further details and samples are located on the [github docs].

## Capabilities
The package allows you to convert any Unity scene into a learning environment and train character behaviors using a variety of machine-learning algorithms. Additionally, it allows you to embed these trained behaviors back into Unity scenes to control your characters. More specifically, the package provides the following core functionalities:

* Define Agents: entities, or characters, whose behavior will be learned. Agents are entities that generate observations (through sensors), take actions, and receive rewards from the environment.
* Define Behaviors: entities that specify how an agent should act. Multiple agents can share the same Behavior and a scene may have multiple Behaviors.
* Record demonstrations: To show the behaviors of an agent within the Editor. You can use demonstrations to help train a behavior for that agent.
* Embed a trained behavior (aka: run your ML model) in the scene via the [Unity Inference Engine]. Embedded behaviors allow you to switch an Agent between learning and inference.

## Special Notes
Note that the ML-Agents package does not contain the machine learning algorithms for training behaviors. The ML-Agents package only supports instrumenting a Unity scene, setting it up for training, and then embedding the trained model back into your Unity scene. The machine learning algorithms that orchestrate training are part of the companion [python package].

## Package contents

The following table describes the package folder structure:

| **Location**           | **Description**                                                         |
| ---------------------- | ----------------------------------------------------------------------- |
| _Documentation~_       | Contains the documentation for the Unity package.                       |
| _Editor_               | Contains utilities for Editor windows and drawers.                      |
| _Plugins_              | Contains third-party DLLs.                                              |
| _Runtime_              | Contains core C# APIs for integrating ML-Agents into your Unity scene.  |
| _Runtime/Integrations_ | Contains utilities for integrating ML-Agents into specific game genres. |
| _Tests_                | Contains the unit tests for the package.                                |

<a name="Installation"></a>

## Installation
To add the ML-Agents package to a Unity project:

* Create a new Unity project with Unity 6000.0 (or later) or open an existing one.
* To open the Package Manager, navigate to Window > Package Manager.
* Click + and select Add package by name...
* Enter com.unity.ml-agents
*Click Add to add the package to your project.

To install the companion Python package to enable training behaviors, follow the [installation instructions] on our [GitHub repository].

## Advanced Features

### Custom Grid Sensors

Grid Sensor provides a 2D observation that detects objects around an agent from a top-down view. Compared to RayCasts, it receives a full observation in a grid area without gaps, and the detection is not blocked by objects around the agents. This gives a more granular view while requiring a higher usage of compute resources.

One extra feature with Grid Sensors is that you can derive from the Grid Sensor base class to collect custom data besides the object tags, to include custom attributes as observations. This allows more flexibility for the use of GridSensor.

#### Creating Custom Grid Sensors
To create a custom grid sensor, you'll need to derive from two classes: `GridSensorBase` and `GridSensorComponent`.

##### Deriving from `GridSensorBase`
This is the implementation of your sensor. This defines how your sensor process detected colliders,
what the data looks like, and how the observations are constructed from the detected objects.
Consider overriding the following methods depending on your use case:
* `protected virtual int GetCellObservationSize()`: Return the observation size per cell. Default to `1`.
* `protected virtual void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)`: Constructs observations from the detected object. The input provides the detected GameObject and the index of its tag (0-indexed). The observations should be written to the given `dataBuffer` and the buffer size is defined in `GetCellObservationSize()`. This data will be gathered from each cell and sent to the trainer as observation.
* `protected virtual bool IsDataNormalized()`: Return whether the observation is normalized to 0~1. This affects whether you're able to use compressed observations as compressed data only supports normalized data. Return `true` if all the values written in `GetObjectData` are within the range of (0, 1), otherwise return `false`. Default to `false`.

    There might be cases when your data is not in the range of (0, 1) but you still wish to use compressed data to speed up training. If your data is naturally bounded within a range, normalize your data first to the possible range and fill the buffer with normalized data. For example, since the angle of rotation is bounded within `0 ~ 360`, record an angle `x` as `x/360` instead of `x`. If your data value is not bounded (position, velocity, etc.), consider setting a reasonable min/max value and use that to normalize your data.
* `protected internal virtual ProcessCollidersMethod GetProcessCollidersMethod()`: Return the method to process colliders detected in a cell. This defines the sensor behavior when multiple objects with detectable tags are detected within a cell.
Currently two methods are provided:
  * `ProcessCollidersMethod.ProcessClosestColliders` (Default): Process the closest collider to the agent. In this case each cell's data is represented by one object.
  * `ProcessCollidersMethod.ProcessAllColliders`: Process all detected colliders. This is useful when the data from each cell is additive, for instance, the count of detected objects in a cell. When using this option, the input `dataBuffer` in `GetObjectData()` will contain processed data from other colliders detected in the cell. You'll more likely want to add/subtract values from the buffer instead of overwrite it completely.

##### Deriving from `GridSensorComponent`
To create your sensor, you need to override the sensor component and add your sensor to the creation.
Specifically, you need to override `GetGridSensors()` and return an array of grid sensors you want to use in the component.
It can be used to create multiple different customized grid sensors, or you can also include the ones provided in our package (listed in the next section).

Example:
```csharp
public class CustomGridSensorComponent : GridSensorComponent
{
    protected override GridSensorBase[] GetGridSensors()
    {
        return new GridSensorBase[] { new CustomGridSensor(...)};
    }
}
```

#### Grid Sensor Types
Here we list out two types of grid sensor provided in the package: `OneHotGridSensor` and `CountingGridSensor`.
Their implementations are also a good reference for making you own ones.

##### OneHotGridSensor
This is the default sensor used by `GridSensorComponent`. It detects objects with detectable tags and the observation is the one-hot representation of the detected tag index.

The implementation of the sensor is defined as following:
* `GetCellObservationSize()`: `detectableTags.Length`
* `IsDataNormalized()`: `true`
* `ProcessCollidersMethod()`: `ProcessCollidersMethod.ProcessClosestColliders`
* `GetObjectData()`:

```csharp
protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
{
    dataBuffer[tagIndex] = 1;
}
```

##### CountingGridSensor
This is an example of using all colliders detected in a cell. It counts the number of objects detected for each detectable tag. The sensor cannot be used with data compression.

The implementation of the sensor is defined as following:
* `GetCellObservationSize()`: `detectableTags.Length`
* `IsDataNormalized()`: `false`
* `ProcessCollidersMethod()`: `ProcessCollidersMethod.ProcessAllColliders`
* `GetObjectData()`:

```csharp
protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
{
    dataBuffer[tagIndex] += 1;
}
```

### Input System Integration

The ML-Agents package integrates with the [Input System Package](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.1/manual/QuickStartGuide.html) through the `InputActuatorComponent`. This component sets up an action space for your `Agent` based on an `InputActionAsset` that is referenced by the `IInputActionAssetProvider` interface, or the `PlayerInput` component that may be living on your player controlled `Agent`. This means that if you have code outside of your agent that handles input, you will not need to implement the Heuristic function in agent as well. The `InputActuatorComponent` will handle this for you. You can now train and run inference on `Agents` with an action space defined by an `InputActionAsset`.

Take a look at how we have implemented the C# code in the example Input Integration scene (located under Project/Assets/ML-Agents/Examples/PushBlockWithInput/). Once you have some familiarity, then the next step would be to add the InputActuatorComponent to your player Agent. The example we have implemented uses C# Events to send information from the Input System.

#### Getting Started with Input System Integration
1. Add the `com.unity.inputsystem` version 1.1.0-preview.3 or later to your project via the Package Manager window.
2. If you have already setup an InputActionAsset skip to Step 3, otherwise follow these sub steps:
    1. Create an InputActionAsset to allow your Agent to be controlled by the Input System.
    2. Handle the events from the Input System where you normally would (i.e. a script external to your Agent class).
3. Add the InputSystemActuatorComponent to the GameObject that has the `PlayerInput` and `Agent` components attached.

Additionally, see below for additional technical specifications on the C# code for the InputActuatorComponent.
#### Technical Specifications

##### `IInputActionsAssetProvider` Interface
The `InputActuatorComponent` searches for a `Component` that implements
`IInputActionAssetProvider` on the `GameObject` they both are attached to. It is important to note
that if multiple `Components` on your `GameObject` need to access an `InputActionAsset` to handle events,
they will need to share the same instance of the `InputActionAsset` that is returned from the
`IInputActionAssetProvider`.

##### `InputActuatorComponent` Class
The `InputActuatorComponent` is the bridge between ML-Agents and the Input System. It allows ML-Agents to:
* create an `ActionSpec` for your Agent based on an `InputActionAsset` that comes from an
`IInputActionAssetProvider`.
* send simulated input from a training process or a neural network
* let developers keep their input handling code in one place

This is accomplished by adding the `InputActuatorComponent` to an Agent which already has the PlayerInput component attached.

## Known Limitations

### Training

Training is limited to the Unity Editor and Standalone builds on Windows, MacOS,
and Linux with the Mono scripting backend. Currently, training does not work
with the IL2CPP scripting backend. Your environment will default to inference
mode if training is not supported or is not currently running.

### Inference

Inference is executed via [Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) on the end-user device. Therefore, it is subject to the performance limitations of the end-user CPU or GPU. Also, only models created with our trainers are supported for running ML-Agents with a neural network behavior.

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time. The Academy (the sentinel that controls the stepping of the game to make sure everything is synchronized, from collection of observations to applying actions generated from policy inference to the agent) also moves in time with `FixedUpdate()` rather than `Update()`, so game behavior implemented in Update() may be out of sync with the agent decision-making.  See [Execution Order of Event Functions] for more information.

You can control the frequency of Academy stepping by calling `Academy.Instance.DisableAutomaticStepping()`, and then calling `Academy.Instance.EnvironmentStep()`.

### Input System Integration

 For the `InputActuatorComponent`
  - Limited implementation of `InputControls`
  - No way to customize the action space of the `InputActuatorComponent`

## Additional Resources

* [GitHub repository]
* [Unity Discussions]
* [Discord]
* [Website]

[github docs]: https://unity-technologies.github.io/ml-agents/
[installation instructions]: https://github.com/Unity-Technologies/ml-agents/blob/release_22_docs/docs/Installation.md
[Unity Inference Engine]: https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/index.html
[python package]: https://github.com/Unity-Technologies/ml-agents
[GitHub repository]: https://github.com/Unity-Technologies/ml-agents
[Execution Order of Event Functions]: https://docs.unity3d.com/Manual/ExecutionOrder.html
[Unity Discussions]: https://discussions.unity.com/tag/ml-agents
[Discord]: https://discord.com/channels/489222168727519232/1202574086115557446
[Website]: https://unity-technologies.github.io/ml-agents/

