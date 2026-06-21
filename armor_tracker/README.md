# armor_tracker

Consumes detected armor plates from `armor_detector` and runs an Extended Kalman Filter to track targets across frames, predict through occlusion, and output a full state estimate.

## nodes

**ArmorTrackerNode** subscribes to `/detector/armors` and `/tf`, transforms armor positions into the inertial frame (gimbal center origin, IMU yaw-aligned X axis), and publishes to `/tracker/target`.

## state machine

| State | Description |
|---|---|
| `DETECTING` | Target spotted, waiting for enough consecutive frames to confirm |
| `TRACKING` | Actively tracking, EKF updating every frame |
| `TEMP_LOST` | Target disappeared, EKF predicting position |
| `LOST` | Lost for too long, tracker reset |

On init, the tracker picks the armor closest to the image center as the initial target.

## EKF

State vector (9D):

```
x = [xc, vx, yc, vy, z, vz, yaw, vyaw, r]
```

- `xc, yc, z` — robot center position in inertial frame
- `vx, vy, vz` — center velocity
- `yaw, vyaw` — robot yaw and angular velocity
- `r` — radius from robot center to armor plate

Observation maps state back to armor position:

```
xa = xc - r * cos(yaw)
ya = yc - r * sin(yaw)
```

This lets the filter stay locked on the robot even when it rotates and a different armor plate comes into view.

## parameters

| Parameter | Default | Description |
|---|---|---|
| `tracker.max_match_distance` | 0.15 | Max 3D distance (m) to associate a detection with the tracked target |
| `tracker.max_match_yaw_diff` | 1.0 | Max yaw difference (rad) for association |
| `tracker.tracking_thres` | 5 | Frames required to transition to TRACKING |
| `tracker.lost_time_thres` | 0.3 | Seconds before transitioning to LOST |
| `ekf.sigma2_q_xyz` | 20.0 | Process noise for position/velocity |
| `ekf.sigma2_q_yaw` | 100.0 | Process noise for yaw |
| `ekf.sigma2_q_r` | 800.0 | Process noise for radius |
| `ekf.r_xyz_factor` | 0.05 | Measurement noise scale for position |
| `ekf.r_yaw` | 0.02 | Measurement noise for yaw |
