# robo-auto-aim

Real-time auto-aiming system for RoboMaster competitive robotics. Takes a raw camera feed and outputs a tracked 3D target position with velocity prediction — fast enough to feed a ballistic solver and actually hit moving targets.

Built in C++14 on ROS2 Humble. Detection runs classical threshold + contour light-strip matching by default, and switches to a trained YOLOv8 model when one is present (see [Training](#training) — no model ships with the repo). Either way an MLP classifies the plate number. The tracking stage runs a 9-state Extended Kalman Filter that handles robot spin, temporary occlusion, and armor plate switching.

---

## Pipeline

```
Camera → YOLOv8 Detector → MLP Classifier → PnP Solver → EKF Tracker → Target State
```

**Detection** — YOLOv8n finds armor plates in each frame (small vs. large), or classical threshold + contour detection does if no ONNX model is present. A perspective-warped ROI is passed to an MLP to classify which robot number (1–5, outpost, guard, base).

Both paths gate on team color via `detect_color`. The classical path reads it from the light-strip contours; because the YOLO model classifies plate *size* only, that path samples the strip bands at each box edge instead. A plate too dim to call is reported as unknown and dropped — the detector will not engage a plate it cannot attribute to the enemy.

**Pose estimation** — OpenCV's `solvePnP` uses known armor dimensions (135×55 mm small, 225×55 mm large) and the camera calibration matrix to get a 3D position and orientation for each detection.

**Tracking** — Extended Kalman Filter with a circular motion observation model. Tracks robot center position rather than armor position directly, so the filter stays locked even when the robot spins and a different armor plate comes into view.

---

## Packages

| Package | Description |
|---|---|
| `armor_detector` | YOLOv8 inference, number classification, PnP pose solving |
| `armor_tracker` | EKF tracker, state machine, TF2 coordinate transforms |
| `auto_aim_interfaces` | Custom ROS2 message definitions |

---

## Build

```bash
cd your_ros_ws/src
git clone https://github.com/scotthnguyen/robo-auto-aim.git

cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to sn_auto_aim
```

Requires ROS2 Humble on Ubuntu 22.04.

---

## Run

Both nodes are `rclcpp` components; the launch file composes them into a single
container so armor messages pass between them via intra-process comms.

```bash
source install/setup.bash
ros2 launch sn_auto_aim sn_auto_aim.launch.py
```

The tracker transforms armor poses into `target_frame` (default `odom`), so a
TF from your camera's optical frame to that frame has to be published. Point a
camera driver at `/image_raw` and `/camera_info` and the pipeline starts on the
first frame.

---

## Tests

```bash
colcon test --packages-select armor_detector armor_tracker \
  --event-handlers console_cohesion+ --return-code-on-test-failure
colcon test-result --all --verbose
```

| Suite | Covers |
|---|---|
| `test_detector` | Light-strip segmentation, armor pairing, small/large classification, color gating |
| `test_detector_images` | The above against the real captures in `docs/`, across the usable `binary_thres` range |
| `test_yolo_detector` | Team-color sampling from strip bands, and the enemy-only filter applied to YOLO detections |
| `test_pnp_solver` | Pose recovery against known armor geometry, reprojection round-trip, distortion-vector lengths |
| `test_number_cls` | MLP classifier forward-pass benchmark |
| `test_node_startup` | Detector node constructs and tears down |
| `test_kalman_filter` | EKF predict/update on a constant-velocity model |
| `test_tracker` | Tracker state machine, data association, armor-jump handling, radius clamping |

---

## Training

### YOLOv8 armor detector

```bash
pip install ultralytics
cd train_yolo
# put images in data/images/train|val and labels in data/labels/train|val
python train.py --epochs 100 --device 0
```

Exports `armor.onnx` directly to `armor_detector/model/`. Rebuild and the node picks it up automatically.

Dataset format: YOLO bounding box labels, 2 classes — `small_armor` (0), `large_armor` (1).

### MLP number classifier

```bash
pip install torch opencv-python
cd train_classifier
# put images in data/<class>/ folders (1 2 3 4 5 outpost guard base negative)
python train.py --epochs 50
```

Exports `mlp.onnx` and `label.txt` to `armor_detector/model/`.

---

## EKF State Model

State vector: `[xc, vx, yc, vy, z, vz, yaw, v_yaw, r]`

The observation function maps robot center + radius + yaw to the observed armor position:

```
xa = xc - r·cos(yaw)
ya = yc - r·sin(yaw)
```

This lets the filter maintain a smooth estimate of robot center even as different armor plates rotate into view. Radius `r` is part of the state, so the filter self-corrects as it observes more of the robot's geometry.

State machine: `LOST → DETECTING → TRACKING → TEMP_LOST`

---

## ROS2 Topics

| Topic | Type | Direction |
|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | Input |
| `/camera_info` | `sensor_msgs/CameraInfo` | Input |
| `/detector/armors` | `auto_aim_interfaces/Armors` | Detector → Tracker |
| `/tracker/target` | `auto_aim_interfaces/Target` | Output |

---

## Parameters

Key tunable params (set via ROS2 launch or `ros2 param set`):

| Parameter | Default | Description |
|---|---|---|
| `yolo_confidence` | 0.5 | YOLO detection confidence threshold |
| `yolo_nms_threshold` | 0.45 | NMS IoU threshold |
| `classifier_threshold` | 0.7 | MLP confidence cutoff |
| `tracker.max_match_distance` | 0.15 m | Max position diff for EKF data association |
| `tracker.lost_time_thres` | 0.3 s | Time before TEMP_LOST → LOST transition |
| `ekf.sigma2_q_xyz` | 20.0 | Process noise — position |
| `ekf.sigma2_q_yaw` | 100.0 | Process noise — yaw |
