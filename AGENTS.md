# Autonomous Drone Project Rules

## Project Context
- This project uses computer vision to command an ArduPilot-based drone to follow a person.
- Development happens in two phases:
  1. ArduPilot SITL + Gazebo simulation.
  2. Real drone deployment with the vision stack on a Radxa Dragon Q6A and the flight controller connected over UART.
- Safety has priority over tracking performance, convenience, or aggressiveness.

## Coding Rules
- Write Python code.
- Keep the code modular, but do not collapse everything into one file and do not create unnecessary files.
- Use clear function and class docstrings describing purpose, parameters, return values, and important side effects.
- Do not over-comment. Add comments only where they reduce real ambiguity.
- Prefer explicit types, small functions, and straightforward control flow.
- Keep tunable values centralized in a configuration module or dataclass rather than scattering literals through the code.
- Use consistent physical units in code and docs. Prefer meters, seconds, radians, and meters/second internally.

## Flight Control Rules
- Assume ArduPilot `GUIDED` mode for autonomous following unless the user requests otherwise.
- The companion computer must never fight the pilot for control.
- Autonomous follow behavior must only run when an explicit enable condition is true, such as a dedicated RC switch and a valid autopilot mode/state.
- When autonomous follow is disabled, the system must immediately stop sending follow commands and yield back to pilot control.
- Prefer MAVLink body-frame velocity commands over direct attitude control for the first implementation.
- Do not command aggressive motion. Clamp commanded lateral speed, forward speed, vertical speed, yaw rate, and acceleration.
- Rate-limit command changes so the drone cannot jerk due to noisy detections.
- Maintain a fixed altitude by default, with altitude as a configurable parameter.
- Maintain a configurable minimum horizontal stand-off distance from the person. Default planning value: about 13 feet horizontal and about 8 feet altitude.
- Treat camera mount angle as part of the control model. Do not assume the camera optical axis is level with the world frame.

## Safety Rules
- Build for fail-safe behavior first and tracking performance second.
- On target loss, stale detections, low-confidence tracking, MAVLink timeout, invalid vehicle state, or conflicting pilot state:
  1. command zero horizontal velocity,
  2. avoid sudden yaw or altitude changes,
  3. transition to a safe hold behavior such as hover or loiter if appropriate.
- Never continue motion on stale vision data.
- Require bounded confidence thresholds and timeout thresholds for all perception-driven actions.
- Keep geofence, RTL, battery, EKF, and ArduPilot safety features enabled and compatible with the companion-computer logic.
- Any behavior that could reduce pilot authority must be opt-in and easy to disable from the transmitter.

## Tracking and Vision Rules
- Favor solutions that meet real-time constraints on the Radxa Dragon Q6A.
- Prefer a lightweight tracker unless a heavier re-identification model clearly improves safety or reliability enough to justify the latency.
- Keep detector, tracker, and controller interfaces separate so models can be swapped without rewriting the whole stack.
- Expose thresholds for detection confidence, tracking confidence, target size, target loss timeout, and control deadbands as configuration.

## Testing Rules
- Validate new control logic in SITL before any real flight testing.
- Add focused tests for math-heavy or safety-critical logic such as coordinate transforms, command clamps, and lost-target behavior.
- For real-drone features, include a simulation-first path and clearly mark any assumptions that still require bench or flight validation.
