# C.R.A.T.E.R. Rover Simulation

## Abstract

C.R.A.T.E.R. (Constrained Reconnaissance, Autonomy & Telemetry in Extreme
Regions) is a Webots-based simulation of a Pioneer 3-AT rover exploring a
crater-like environment. The project combines simulated perception, rover
motion estimation, threat reporting, obstacle-aware navigation, telemetry
transmission, Reed-Solomon loss recovery, a live dashboard, and SQLite
mission logging.

The system is intentionally split into two related paths:

1. **Webots simulation path:** the rover reads simulated sensors, estimates its
   pose, detects nearby LiDAR threats, chooses a safe velocity, and publishes
   telemetry to a Webots dashboard bridge.
2. **Edge/radio path:** the ONNX vision detector writes detected hazards to
   `hazards.json`; the telemetry transmitter packs them into radio packets and
   the Streamlit mission-control dashboard receives, displays, acknowledges, and
   stores them.

This separation makes it possible to test rover motion and obstacle avoidance
inside Webots while also testing constrained radio communication and ground
station persistence.

## What the robot is made with

### Simulation and software

- **Webots R2025a** for the world, physics, sensors, and Pioneer 3-AT model.
- **Python** for the controllers, navigation policy, detector, radio protocol,
  dashboards, and database writer.
- **NumPy** for sensor arrays, image/depth data, and numerical processing.
- **OpenCV** for image preparation, resizing, color conversion, and detection
  post-processing.
- **ONNX Runtime** for CPU inference using `yolov8n.onnx`.
- **Streamlit, Plotly, and pandas** for the radio mission-control dashboard.
- **SQLite** through Python's built-in `sqlite3` module for telemetry history.
- **Reed-Solomon coding** through the `reedsolo` package for packet recovery.

### Rover hardware model

The simulated Pioneer 3-AT has:

| Sensor or actuator | Configuration | Use |
|---|---:|---|
| RGB camera | 640 × 480, 1.2 rad FOV | Vision input |
| 3D LiDAR | 256 horizontal samples, 16 layers, 180° FOV | Obstacles, terrain, threats |
| Inertial unit | Roll, pitch, yaw | Orientation |
| Accelerometer | Three axes | Motion data |
| Gyroscope | Three axes | Angular motion and heading support |
| GPS | Three-dimensional position | Simulation reference position |
| Wheel encoders | Left and right position sensors | Wheel odometry |
| Wheel motors | Differential drive | Linear and angular motion |

The Webots basic time step is 16 ms. The camera and LiDAR are sampled every
four controller steps, approximately 15.6 Hz. The IMU, accelerometer,
gyroscope, GPS, and encoders are updated every simulation step.

## Data produced by the rover

After initialization, `Rover.observation()` provides:

| Field | Shape/type | Description |
|---|---|---|
| `camera` | `numpy.ndarray`, `(480, 640, 3)` | RGB camera image |
| `lidar` | `numpy.ndarray`, `(16, 256)` | LiDAR ranges in metres |
| `imu` | `numpy.ndarray`, `(9,)` | Roll, pitch, yaw, acceleration, angular velocity |
| `encoders` | `numpy.ndarray` | Wheel encoder angles in radians |
| `odom` | `numpy.ndarray`, `(3,)` | Estimated `(x, y, yaw)` pose |
| `position` | dictionary | Position and yaw signal for telemetry |
| `threats` | list | LiDAR threats classified by sector and distance |
| `time` | float | Webots simulation time in seconds |

The telemetry payload contains the pose, position, rover status, and threats.
Threat records include:

```json
{
  "type": "obstacle",
  "sector": "front",
  "distance": 0.82,
  "bearing": 0.0,
  "x": 1.2,
  "y": -0.4,
  "severity": "critical"
}
```

The current LiDAR threat sectors are `left`, `front`, and `right`. Threats
closer than 1 metre are reported as `critical`; threats below 3 metres are
reported as `warning`.

## Threat-detection AI

### Base model

The edge detector uses **YOLOv8n**, the nano-sized YOLOv8 model, exported to
ONNX and loaded by [edge_detector.py](./edge_detector.py) with
`onnxruntime`.

The model was selected because it provides a practical balance between:

- low inference cost on CPU hardware;
- small model size suitable for an edge computer;
- fast inference for repeated camera frames;
- a standard YOLO detection output that is straightforward to post-process;
- compatibility with later fine-tuning for rover-specific classes.

The detector uses 640 × 640 input tensors, confidence threshold `0.35`, IoU
threshold `0.45`, and class-wise non-maximum suppression. The ONNX model is
configured for the CPU execution provider.

### Important model limitation

`yolov8n.onnx` is a base COCO-style model unless it has been replaced with a
custom trained export. The project currently maps class IDs to names in the
dashboard:

```text
0 -> rock
1 -> crater
2 -> ice_deposit
```

Those names are meaningful only when the model has been trained with the same
class ordering. With an unmodified COCO model, the class IDs do not represent
these mission-specific classes. A production version should fine-tune YOLOv8n
on labelled rover imagery and export the resulting model to ONNX.

### Vision and LiDAR roles

The vision detector and Webots LiDAR threat detector are complementary:

- **YOLO/ONNX:** identifies visual objects in images and writes hazard records
  with class, confidence, position, detection time, and uncertainty.
- **LiDAR:** provides immediate geometric obstacle distances for navigation and
  live Webots threat reporting.

The current Webots safety policy uses LiDAR directly. It does not yet feed
YOLO detections into the motion controller.

## Coordinate system and position

The rover uses a `base_link` coordinate frame:

- `x`: forward;
- `y`: left;
- `z`: up.

The sensor mounts are:

```text
LiDAR:  (0.00, 0.00, 0.45) m
Camera: (0.25, 0.00, 0.30) m
```

The generated world starts the rover inside the crater at approximately
`(x=0.0, y=-25.0)` metres. The main crater radius is approximately 42 metres.

The rover exposes three position-related quantities:

1. **Wheel odometry:** encoder angle changes are converted to wheel travel.
   Average wheel travel gives forward displacement, and pitch compensation
   projects the displacement onto the horizontal plane.
2. **IMU orientation:** roll, pitch, and yaw are read from the inertial unit
   and used to orient the odometry update.
3. **GPS reference:** GPS displacement from the initial reading provides
   simulation ground truth. The rover compares this position with odometry to
   calculate odometry error.

GPS is currently an evaluation/reference source in the main controller. It is
not yet a complete production-grade sensor-fusion correction step.

## Odometry, redundancy, and simulation noise

### Odometry method

The current odometry implementation:

1. reads the left and right wheel encoder angles;
2. calculates the change since the previous update;
3. converts wheel rotation to distance using the wheel radius (`0.11 m`);
4. averages left and right travel to estimate forward motion;
5. compensates for pitch using `distance × cos(pitch)`;
6. projects the horizontal distance into world coordinates using IMU yaw;
7. compares the estimated pose against GPS displacement.

The differential-drive parameters include a wheel track of `0.394 m` and motor
limits of `6.4 rad/s`.

### Redundancy

The simulation deliberately provides multiple sources for important quantities:

- wheel encoders provide incremental displacement;
- the IMU provides orientation;
- GPS provides an independent position reference;
- LiDAR provides an independent geometric view of nearby obstacles;
- the camera and neural detector provide visual hazard information.

The communication path also has redundancy: each DATA packet is transmitted
with Reed-Solomon parity and can be retransmitted using acknowledgements.

### Noise and faults

The optional sensor model in
[sensor_models.py](./controllers/my_controller/sensor_models.py) defines:

- encoder quantization at 2048 counts per revolution;
- encoder noise of 0.3 counts;
- gyro initial bias, bias random walk, and white noise;
- accelerometer bias and white noise;
- roll/pitch noise of 0.002 radians;
- LiDAR absolute noise of 0.02 m plus 1% range noise;
- LiDAR dropout probability of 2%;
- camera pixel noise with standard deviation 3 gray levels.

The same module includes test fault switches for camera blindness and LiDAR
blindness, plus hooks for encoder freezing, encoder scale errors, and bias
changes. The active rover controller should be checked before experiments to
confirm which optional noise wrappers are enabled in that run.

## Navigation and obstacle avoidance

[policy.py](./policy.py) uses the LiDAR range image to produce linear and
angular velocity commands:

- stop and turn when the front distance is below 1 metre;
- slow down when an obstacle is below 2.2 metres;
- turn toward the side with more free space;
- steer away from close left or right obstacles;
- continue forward at a controlled speed when the path is clear.

This is a reactive local-avoidance policy, not a global planner. It does not
yet provide mapping, waypoint planning, guaranteed obstacle completeness, or
formal collision avoidance. Its behaviour should therefore be evaluated over
multiple random seeds and terrain configurations.

## Communication protocol and Reed-Solomon recovery

### Protocol path

The radio path is:

```text
edge_detector.py
        ↓
hazards.json
        ↓
telemetry_tx.py
        ↓ UDP 9002
dashboard.py
        ↓
crater_telemetry.db
```

[link_protocol.py](./link_protocol.py) defines the shared wire format for the
transmitter and receiver. DATA packets include:

- magic value and packet kind;
- session identifier;
- sequence number;
- rover status;
- hazard records;
- CRC-8 checksum.

Hazard positions use centimetres on the wire, confidence uses an 8-bit value,
uncertainty uses decimetres, and detection age uses centiseconds.

### Reed-Solomon protection

Each DATA packet is divided into seven data shards and three parity shards:

```text
7 data shards + 3 parity shards = 10 total shards
```

The receiver can reconstruct the original packet when up to three of the ten
shards are lost or corrupted. Every shard is protected by CRC-8 before
reassembly.

If loss exceeds the Reed-Solomon correction capacity, selective-repeat ARQ
provides another layer of recovery:

1. the ground station sends an ACK containing a base hazard ID and bitmap;
2. the rover keeps unacknowledged hazards pending;
3. pending hazards are retransmitted with exponential backoff;
4. the ground station de-duplicates hazards that arrive again.

The default test configuration simulates 30% packet loss. This is a test
condition, not a guarantee that an arbitrary real radio link will lose no data.

## Dashboards

There are two dashboards for two telemetry paths.

### Webots dashboard

Run:

```powershell
.\venv\Scripts\python.exe controllers\dashboard\dashboard_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The Webots world starts the internal
[dashboard controller](./controllers/dashboard/dashboard.py) automatically.
It receives rover telemetry from the Webots `Emitter` and forwards it to
`dashboard_app.py` on UDP port 9000.

The Webots dashboard shows:

- rover position and yaw;
- trajectory trail;
- telemetry connection state;
- LiDAR threat sectors;
- obstacle distances and severity;
- odometry status and diagnostic signals.

### Streamlit mission control

Run:

```powershell
python -m streamlit run dashboard.py
```

This dashboard listens on UDP port 9002 and displays the edge/radio hazards,
packet health, rover status, uncertainty circles, and data aging.

## SQLite telemetry database

The Streamlit ground station saves every valid reconstructed DATA packet to:

```text
crater_telemetry.db
```

The database contains:

- `packets`: receive time, session, sequence, packet size, and raw packet bytes;
- `hazards`: decoded hazard records, confidence, uncertainty, detection age,
  coordinates, and detection time;
- `rover_status`: rover x/y/yaw and odometry error for each packet.

The database is created automatically. To use a different path:

```powershell
$env:CRATER_DB = "history\mission.sqlite"
python -m streamlit run dashboard.py
```

Runtime databases and telemetry files are ignored by Git.

## Project structure

```text
.
├── worlds/
│   └── crater_world.wbt
├── controllers/
│   ├── dashboard/
│   │   ├── dashboard.py
│   │   └── dashboard_app.py
│   └── my_controller/
│       ├── my_controller.py
│       ├── rover.py
│       ├── odometry.py
│       ├── frames.py
│       └── sensor_models.py
├── create_creator.py
├── edge_detector.py
├── link_protocol.py
├── policy.py
├── telemetry_database.py
├── telemetry_tx.py
├── dashboard.py
├── requirements.txt
└── run_live.bat
```

## Rebuilding the crater world

The world generator creates the terrain and writes the Webots world to
`worlds/crater_world.wbt`:

```powershell
.\venv\Scripts\python.exe create_creator.py
```

The generated world includes the Pioneer 3-AT, the rover sensors, the
telemetry emitter, and the dashboard receiver.

## Running the Webots simulation

1. Install Webots R2025a and the Python dependencies.
2. Generate the world if required:

   ```powershell
   .\venv\Scripts\python.exe create_creator.py
   ```

3. Start the Webots dashboard:

   ```powershell
   .\venv\Scripts\python.exe controllers\dashboard\dashboard_app.py
   ```

4. Open `worlds/crater_world.wbt` in Webots and start the simulation.
5. Open `http://127.0.0.1:8000`.

The Webots simulation and the Streamlit radio dashboard can be run
independently.

## Dependencies

Install the Python dependencies with:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Webots `controller` module is supplied by the Webots installation and is
not a normal PyPI dependency.

## Limitations and next steps

- Fine-tune YOLOv8n on labelled rover hazard data.
- Connect visual detections to the Webots navigation policy.
- Replace reactive avoidance with a local planner and map.
- Add formal collision and route-completion metrics.
- Integrate GPS, IMU, and wheel odometry into a documented fused estimator.
- Add automated Webots regression tests for threats, navigation, and telemetry.
