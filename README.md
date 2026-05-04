# Autonomous Drone Follower

## Overview

This project detects a person in the camera feed, keeps that person selected across frames, and sends MAVLink follow commands to an ArduPilot-based drone in `GUIDED` mode or conservative stick inputs in `ALT_HOLD`.

The pipeline is:

1. YOLO detects people in each frame.
2. ByteTrack links detections across frames and helps keep a stable target ID.
3. The target selector keeps the same person locked when tracking blips or IDs shift.
4. An optional YOLO pose model can sample the selected person every few seconds and detect a conservative right-hand-raised gesture.
5. The follow controller tries to keep the person near the center of the frame and maintain a configurable stand-off distance using body-frame velocity and yaw-rate commands.
6. When enabled, a raised right hand triggers one bounded orbit around the selected person, then normal follow resumes.
7. MAVLink gating ensures both follow and orbit commands only run when the required vehicle state allows it.

## Project Layout

- [src/autonomous_drone/app.py](src/autonomous_drone/app.py) - CLI entrypoint
- [src/autonomous_drone/control.py](src/autonomous_drone/control.py) - follow controller
- [src/autonomous_drone/mavlink.py](src/autonomous_drone/mavlink.py) - ArduPilot MAVLink client and follow gate logic
- [src/autonomous_drone/perception.py](src/autonomous_drone/perception.py) - YOLO, ByteTrack, and target selection
- [src/autonomous_drone/config.py](src/autonomous_drone/config.py) - app configuration dataclasses
- [configs/sitl_follow.example.json](configs/sitl_follow.example.json) - example SITL config
- [configs/sitl_follow.local.json](configs/sitl_follow.local.json) - local SITL config
- [configs/real_drone_uart.example.json](configs/real_drone_uart.example.json) - example real-drone UART config

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install all dependencies from [requirements.txt](requirements.txt):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Print The Effective Config

```bash
cd src
python -m autonomous_drone.app \
  --config ../configs/sitl_follow.example.json \
  --print-config
```

## Run The Follower

```bash
cd src
python -m autonomous_drone.app \
  --config ../configs/sitl_follow.local.json
```

Useful flags:

- `--transport udp --udp-host 127.0.0.1 --udp-port 14550` selects a structured SITL endpoint
- `--transport serial --serial-device /dev/ttyUSB0 --baud 921600` selects a UART-connected flight controller
- `--skip-rc-gate` lets you test in `GUIDED` mode without needing the RC switch in SITL
- `--dry-run` runs detection, tracking, and control without sending MAVLink commands
- `--enable-guided-nogps-follow` enables the no-GPS `ALT_HOLD` stick-control path
- `--enable-hand-raise-circle` enables the optional pose-triggered one-shot orbit behavior
- `--pose-model /path/to/yolo11n-pose.pt` points the gesture feature at YOLO pose weights
- `--visualize` shows the detection box, target ID, `area_ratio`, and command overlay
- `--log-data` writes per-frame, detection, and tracking CSVs into `logs/`
- `--log-output-dir PATH` changes the CSV output directory from the default `logs/`

Example:

```bash
cd src
python -m autonomous_drone.app \
  --config ../configs/sitl_follow.local.json \
  --log-data \
  --skip-rc-gate \
  --enable-guided-nogps-follow \
  --pose-model /absolute/path/to/yolo11n-pose.pt \
  --visualize
```

## Hand-Raise Orbit Feature

When `pose.hand_raise_circle_enabled` is `true` or `--enable-hand-raise-circle` is passed:

- the app runs YOLO pose estimation on the selected person's crop every `pose.inference_interval_s`
- a conservative right-hand-raised gesture can trigger exactly one orbit
- the orbit uses the existing `GUIDED` body-frame command path, not ArduPilot's native `CIRCLE` mode
- if `GUIDED` is lost, the RC enable switch drops low, the target is lost, or the maneuver times out, the orbit is aborted and autonomy yields immediately

The feature expects a local YOLO pose weights file such as `yolo11n-pose.pt`. The detector and pose models are configured separately:

- `tracking.model_path` is used for person detection
- `pose.model_path` is used for gesture-triggered pose estimation

## Evaluation Logging

When `--log-data` is enabled, each run writes a session directory like `logs/follow-run-YYYYMMDD-HHMMSSZ/`:

- `frames.csv` contains per-frame perception, gate, command, latency, framing-error, and resource-usage data
- `detections.csv` contains raw detector boxes and confidences
- `tracks.csv` contains tracker IDs and tracked boxes

These logs are intended to support:

- system performance analysis such as FPS, latency, CPU, and memory
- tracking stability analysis such as retention time and track ID switches
- drone-follow framing analysis such as center error, target-in-frame accuracy, and distance-proxy error from box area

## Generate Graphs

Pass the per-frame CSV log file to the graph generator:

```bash
cd src
python -m autonomous_drone.generate_graphs logs/follow-run-YYYYMMDD-HHMMSSZ/frames.csv
```

This writes plots and a summary CSV into a sibling `*_plots/` directory.

If you also have hand-labeled ground truth for a run, you can compute label-based detection and tracking metrics too:

```bash
cd src
python -m autonomous_drone.generate_graphs logs/follow-run-YYYYMMDD-HHMMSSZ/frames.csv --ground-truth-csv ../logs/ground_truth.csv
```

Expected ground-truth CSV columns:

- `frame_index`
- `present`
- `x1`, `y1`, `x2`, `y2`
- optional `track_id`

With ground truth, the script computes:

- detection precision, recall, F1, frame-presence accuracy, and mean IoU
- single-target tracking proxies for ID switches, MOTA, IDF1, and retention ratio

Without ground truth, the runtime logs still support evaluation of system performance, target retention, framing accuracy, response latency, and distance-proxy error.

## SITL And Gazebo Commands

### ArduPilot SITL

```bash
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
```

### Gazebo

```bash
gz sim -v4 -r iris_custom.sdf
```

### Gazebo GStreamer

Enable camera streaming:

```bash
gz topic -t /world/iris_runway/model/iris_with_fixed_camera/model/fixed_camera/link/base_link/sensor/camera/image/enable_streaming -m gz.msgs.Boolean -p "data: 1"
```

View the stream:

```bash
gst-launch-1.0 -v udpsrc port=5600 caps='application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264' ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

## Typical Simulation Flow

1. Start Gazebo.
2. Start ArduPilot SITL.
3. Enable the Gazebo camera stream.
4. Confirm the video stream works with `gst-launch-1.0`.
5. Set `tracking.model_path` to your local YOLO weights file.
6. Set `runtime.video_source` in your config to the Gazebo GStreamer pipeline.
7. Run the follower with `python -m autonomous_drone.app --config ...`.

## Real Drone UART Notes

For a direct MAVLink UART link to ArduPilot, set the flight controller serial port to MAVLink2 and match the baud rate used by the companion computer. The ArduPilot companion-computer guide shows `SERIAL2_PROTOCOL = 2` and `SERIAL2_BAUD = 921` for a 921600 baud link:

<https://ardupilot.org/dev/docs/raspberry-pi-via-mavlink.html>

On the Radxa Dragon Q6A 40-pin header, the published pin map shows:

- physical pin `8` as `UART5_TX`
- physical pin `10` as `UART5_RX`
- physical pin `6` as `GND`

Reference:

<https://docs.radxa.com/en/dragon/q6a/hardware-use/pin-gpio>

Example run command:

```bash
cd src
python -m autonomous_drone.app \
  --config ../configs/real_drone_uart.example.json \
  --serial-device /dev/ttyYOUR_UART_DEVICE \
  --baud 921600
```

For no-GPS `ALT_HOLD` follow on a real drone, the app now prefers `RC_CHANNELS_OVERRIDE` for the primary roll, pitch, throttle, and yaw inputs. That tends to be more reliable than `MANUAL_CONTROL` on direct companion-computer links and lets the app release those channels immediately when follow is disabled or the target is lost.

## Notes

- The app expects a local YOLO weights file and does not download one automatically.
- The pose-triggered orbit feature expects a local YOLO pose weights file and is disabled by default.
- Fixed-altitude following is the current mode
- Lateral/Roll motion is still disabled
- The one-shot orbit is implemented in `GUIDED` with bounded velocity and yaw commands instead of switching to ArduPilot `CIRCLE`, because native `CIRCLE` orbits a point fixed when the mode is entered rather than a moving tracked person. Reference: <https://ardupilot.org/copter/docs/circle-mode.html>
