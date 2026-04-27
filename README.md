# Autonomous Drone Follower

## Overview

This project detects a person in the camera feed, keeps that person selected across frames, and sends MAVLink follow commands to an ArduPilot-based drone in `GUIDED` mode.

The pipeline is:

1. YOLO detects people in each frame.
2. ByteTrack links detections across frames and helps keep a stable target ID.
3. The target selector keeps the same person locked when tracking blips or IDs shift.
4. The follow controller tries to keep the person near the center of the frame and maintain a configurable stand-off distance using body-frame velocity and yaw-rate commands.
5. MAVLink gating ensures follow commands only run when the required vehicle state allows it.

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
- `--visualize` shows the detection box, target ID, `area_ratio`, and command overlay

Example:

```bash
cd src
python -m autonomous_drone.app \
  --config ../configs/sitl_follow.local.json \
  --skip-rc-gate \
  --visualize
```

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

## Notes

- The app expects a local YOLO weights file and does not download one automatically.
- Fixed-altitude following is the current mode
- Lateral/Roll motion is still disabled
