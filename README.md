# robo-aim

Real-time armor detection and target tracking for competitive robotics (RoboMaster). Takes a raw camera feed, finds enemy armor plates, classifies which robot it is, and outputs a tracked 3D position with velocity prediction — fast enough to feed a ballistic solver and actually hit moving targets.

## what it does

Two-stage pipeline:

**Detection** — binarizes the image, finds the glowing LED light strips on armor plates via contour analysis, pairs them up geometrically, then runs a perspective warp + MLP neural net to classify which robot (0-9). Solves 3D position using PnP with the camera calibration matrix.

**Tracking** — runs an Extended Kalman Filter on the detector output. Tracks a 9-state vector (center position, velocity, yaw, angular velocity, armor radius). Handles temporary target loss by predicting position until the target reappears. State machine: `LOST -> DETECTING -> TRACKING -> TEMP_LOST`.

## stack

- C++14 / ROS2 Humble
- OpenCV (vision pipeline, PnP solving)
- Eigen (EKF linear algebra)
- ONNX (MLP inference via OpenCV DNN)

## packages

| Package | Description |
|---|---|
| `armor_detector` | Image processing, light detection, number classification, PnP |
| `armor_tracker` | EKF-based target tracking, state machine, TF2 transforms |
| `auto_aim_interfaces` | Custom ROS2 message definitions |

## build

```bash
cd your_ros_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to sn_auto_aim
```

Requires ROS2 Humble on Ubuntu 22.04.

## how the EKF works

State vector: `[xc, vx, yc, vy, z, vz, yaw, vyaw, r]`

The filter models the robot center position with constant velocity. The observation function maps center + radius + yaw back to the observed armor position, which lets the filter stay locked on a robot even when it rotates and a different armor plate becomes visible.

## topics

| Topic | Type | Description |
|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | Input camera feed |
| `/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration |
| `/detector/armors` | `auto_aim_interfaces/Armors` | Detected armor plates with 3D poses |
| `/tracker/target` | `auto_aim_interfaces/Target` | Tracked target with full state estimate |
