# Spatial Memory

This module provide a simple system for the robot to track people and objects, and memorize their locations in the (2D) space 
with respect to an egocentric reference frame.

 _Module Tested against: ***Python 3.8*** and ***YARP 3.4.3***_
 (Update to more recent versions will come soon)

# Explanation

**What does this module do?**

Basically 2 things:
- It tracks objects and/or faces (it depends on what input you give to it in the form of Bounding Boxes)
- It saves the position of each object/person in the "spatial memory". It allocate the object in 5 zones (or bins):
**FAR_RIGHT, RIGHT, CENTER, LEFT, FAR_LEF**T. You can chenge the resolution of the bins if you need.

When I can use it?
The module is useful when the robot have to interact with multiple objects and people. It can be very useful in experiments for group interactions. 

**Why may I need it?**

When interacting with multiple people, you may need a tracker. Differently from a face detector, the face tracker gives a unique ID 
to each face. But be careful! It is not a face recognition!
Given a set of bounding boxes provided by the face detector at the frame i: [detection_1, detection_2, ..., detection_n]
and a set of bounding boxes at the frame i+1: [detection_1, detection_2, ..., detection_n], the face tracker makes associations/matches between the detection boxes across frames.

The tracker used in this module is very simple and based on the Kalman filter. But eventually it can be made more robust by providing feature embeddings.
This is left as future work.

Moreover, since the robot turn its head during interactions and has a limited FOV, it is important to keep persistence of objects
into a temporary memory.

 **Limitations (working on it)**

- The space representation is in 2D and discrete in 5 bins. For now, it is sufficient for our purposes. 
- The kalman-based tracker is not that robust for more dynamic environment. Testing of different trackers are work in progress.


# Dependencies
To run this module you will need an Object/Face Detector.
Currently, I am using the "**objectRecognition**" module, based on yolo8 model and maintained by @Luca Garello.
You can download it from GitLab CognitiveInteraction.


# 📝 Ports
The module reads from an audio stream port and outputs the recognized text.
<details> 
<summary>Click for list of opened ports</summary>

| Port name                     | Description                                                                             |
|-------------------------------|-----------------------------------------------------------------------------------------| 
| /spatialMemory                | RPC handle port used to control the module and set parameters on the go                 |
| /spatialMemory/image:i        | input port to read images from robot cameras                                            |
| /spatialMemory/coord:i        | input port for reading bounding box coordinates from object recognition/ face detection |
| /spatialMemory/image:o        | output port to send image with trackers bounding box + labels                           |
| /spatialMemory/memory_image:o | output port to send image of 2D spatial memory                                          |
| /spatialMemory/icubHeadPose:i | input port to read iCub head position (azimuth angles)                                  |

</details> 

# 🎮 RPC Commands
The translation can be started and stopped sending the following RPC commands to the handler port ``/speech2text``:
<details> 
<summary>Click for list of available commands</summary> 

| Command           | Description                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------|
| start             | TBD                                                                                                   |
| stop              | TBD                                                                                                   |
| set [zone] [role] | set the property "role" to the object present in "zone". This command is specific for addresse module |

</details> 



# 💻 Installation
clone this repo, ideally in the `\CognitiveInteraction` folder.

```bash
cd ../object_recognition

mkdir build && cd build

cmake .. /
make install
```
Remember that with this installation you don't need to manually create a python virtual_env.
It will be handled automatically by CMake files.
An executable will be created and you don't need to build the application every time you do changes. The code will be updated automatically.


## Authors and acknowledgment
This project was developed by Giulia Belgiovine (giulia.belgiovine@iit.it) and Jonas Gonzalez.
It is currently updated and maintained by Giulia Belgiovine

## License


