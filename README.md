# Autonomous Drone Follower

Conservative person-follow controller for ArduPilot SITL and later real-drone deployment.

## What it does
- Uses YOLO for person detection.
- Uses ByteTrack for lightweight multi-frame tracking.
- Sends conservative body-frame velocity and yaw-rate commands through `pymavlink`.
- Gates follow autonomy on `GUIDED` mode plus an RC switch, unless you explicitly bypass the RC gate for simulation.
- Prioritizes low-oscillation behavior through deadbands, low-pass filtering, acquisition hysteresis, and slew-rate limiting.

## Project layout
- [src/autonomous_drone/app.py](src/autonomous_drone/app.py) - CLI entrypoint
- [src/autonomous_drone/control.py](src/autonomous_drone/control.py) - follow controller
- [src/autonomous_drone/mavlink.py](src/autonomous_drone/mavlink.py) - ArduPilot MAVLink client and RC gate
- [src/autonomous_drone/perception.py](src/autonomous_drone/perception.py) - YOLO + ByteTrack pipeline
- [configs/sitl_follow.example.json](configs/sitl_follow.example.json) - simulation-oriented config example

## Run the tests
```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

## Print the effective config
```bash
PYTHONPATH=src ./.venv/bin/python -m autonomous_drone.app \
  --config configs/sitl_follow.example.json \
  --print-config
```

## SITL + Gazebo flow
1. Start ArduPilot SITL.
2. Start Gazebo and expose a camera stream that OpenCV can open directly or through GStreamer.
3. Copy `configs/sitl_follow.example.json` to your own local config and set:
   - `tracking.model_path` to a local YOLO weights file
   - `runtime.video_source` to your Gazebo camera source or GStreamer pipeline
4. Run the follower:

```bash
PYTHONPATH=src ./.venv/bin/python -m autonomous_drone.app \
  --config /path/to/your_sitl_follow.json
```

## Useful simulation flags
- `--skip-rc-gate`: lets you test in `GUIDED` without needing an RC switch path in SITL
- `--dry-run`: runs detection, tracking, and control without sending MAVLink commands
- `--visualize`: shows the detection box, tracking id, and command overlay

Example:

```bash
PYTHONPATH=src ./.venv/bin/python -m autonomous_drone.app \
  --config /path/to/your_sitl_follow.json \
  --skip-rc-gate \
  --visualize
```

## Notes
- The app expects a local YOLO weights file and will not try to download one for you.
- The first controller revision intentionally keeps lateral motion disabled by default to reduce twitchy behavior.
- Fixed-altitude following is the initial mode; vertical chasing is intentionally disabled.
