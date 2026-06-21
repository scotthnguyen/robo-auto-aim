# armor_detector

Handles the full detection pipeline — takes a raw camera image and outputs detected armor plates with 3D positions.

## nodes

**DetectorNode** subscribes to `/camera_info` and `/image_raw`, runs detection, and publishes to `/detector/armors`.

## pipeline

**preprocessImage** — converts to grayscale and applies binary thresholding. Skips HSV-based color filtering because industrial cameras tend to overexpose the LED centers, making R and B channel values roughly equal and color filtering unreliable.

**findLights** — finds contours, fits minimum bounding rectangles, then filters by aspect ratio and tilt angle. Color is determined by summing R vs B pixel values inside the contour.

**matchLights** — pairs light strips by color, filters out pairs that have another light between them, then checks length ratio, center distance, and tilt angle to confirm the pair matches armor plate geometry. Classifies as small (135x55mm) or large (225x55mm).

**NumberClassifier** — warps each armor into a canonical view, crops the digit region, runs Otsu thresholding, and classifies with an MLP (`mlp.onnx`). Input is a flattened 20x28 binary image (560-dim). Two hidden layers, outputs digit 0-9.

**PnPSolver** — solves 3D position from the 4 armor corner points using `cv::solvePnP` with `SOLVEPNP_IPPE` (optimized for coplanar points).

## parameters

| Parameter | Default | Description |
|---|---|---|
| `binary_thres` | 160 | Grayscale threshold for binarization |
| `detect_color` | 0 (RED) | Target color: 0=red, 1=blue |
| `light.min_ratio` | 0.1 | Min width/height ratio for light strips |
| `light.max_ratio` | 0.4 | Max width/height ratio for light strips |
| `light.max_angle` | 40 | Max tilt angle (degrees) |
| `armor.min_light_ratio` | 0.7 | Min length ratio between paired lights |
| `armor.min_small_center_distance` | 0.8 | Small armor min light center distance |
| `armor.max_small_center_distance` | 3.2 | Small armor max light center distance |
| `armor.min_large_center_distance` | 3.2 | Large armor min light center distance |
| `armor.max_large_center_distance` | 5.5 | Large armor max light center distance |
| `armor.max_angle` | 35 | Max armor tilt angle (degrees) |
| `classifier_threshold` | 0.7 | MLP confidence threshold |
| `debug` | false | Publish debug topics |
