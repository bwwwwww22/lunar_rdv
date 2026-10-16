# Simplified 6-DOF SIL Simulator for Lunar Rendezvous & Docking
A closed-loop Python simulator for spacecraft proximity rendezvous
and docking in lunar orbit. The architecture isolates the
vehicle physics (plant) from the flight software (GN&C). Navigation combines
a gyro, star tracker and lidar in a Multiplicative Extended Kalman Filter (MEKF),
and an LQR flight controller closes the loop. Validated via consistency
testing (NEES/NIS), Monte Carlo and sensor-dropout scenarios.

## Architecture

A SIL boundary is enforced between the environment and the flight software
(which sees only noisy sensor data):
    PLANT (truth) → SENSOR (adds noise) → NAV (EKF estimate) → GUIDANCE (static 
    setpoint) → CONTROL (LQR) → ACTUATORS (saturation) → PLANT (next step)

- **Environment / plant** (`src/environment/`): 6-DOF physics. Never sees
  the controller's intent, only the post-saturation applied control.
- **Flight software** (`src/fsw/`): sensors, navigation, guidance, control.
  Never accesses the true state, only noisy measurements.
- **Sim** (`src/sim/`): fixed-rate scheduler and closed-loop runner
- **Analysis** (`src/analysis/`): consistency (NEES/NIS), Monte Carlo, and the
  shared sensor/EKF noise configuration

## Sensors
Three sensors replace the earlier idealized (magical) full-state sensor. Noise is
defined in `default_sensor_config()` (`src/analysis/monte_carlo.py`), which builds
both the sensors and the EKF's R, P0 and Q.

| Sensor | Measures | Rate | Noise (1σ) | Source |
|---|---|---|---|---|
| Gyro | angular rate + bias | every step | ARW 0.5 °/√hr, turn-on bias 5 °/hr, bias random walk 0.5 °/hr/√hr | Honeywell HG1700-AG62 (ARW, bias); random walk is assumed |
| Star tracker | attitude quaternion | 1 Hz | 0.3″ cross-boresight, 1.5″ about boresight | Sodern Hydra datasheet (3σ/3); no systematic bias modeled |
| Lidar | range, azimuth, elevation (body frame) | 1 Hz | 0.02 m, 0.1° | Neptec OPAL precision (range accuracy); angle resolution is assumed |

## Conventions
- **Frames:** translation in Hill/LVLH frame (x=radial, y=along-track,
  z=cross-track); rotation in body frame.
- **Quaternions:** scalar-last `[x, y, z, w]`, JPL convention, passive transform
  from reference to chaser body: `v_body = R(q) @ v_ref`.
- **Units:** meters, seconds, radians, kilograms

## State vector (13) and error state (12)
The full state carries a 4-component quaternion; the EKF operates on a
12-dimensional error state using a 3-parameter small-angle attitude error.

| Index | Symbol | Meaning | Frame | Units |
|-------|--------|---------|-------|-------|
| 0:3   | r      | relative position | Hill | m |
| 3:6   | v      | relative velocity | Hill | m/s |
| 6:10  | q      | attitude quaternion (ref→body, passive) | — | — |
| 10:13 | ω      | angular velocity (plant truth); in the filter this slot holds the gyro-bias estimate b_hat | body | rad/s |

Error state (12): `[δr(3), δv(3), δθ(3), Δb(3)]`, where δθ is a small-angle
attitude error and Δb is the gyro-bias error. Corrections are injected
multiplicatively into the quaternion.

## Dynamics
- **Translation:** Clohessy–Wiltshire equations, lunar mean motion
  `n = √(µ_moon / a³)` for a 100 km circular low lunar orbit. No atmospheric drag.
- **Rotation:** rigid-body Euler equation `I·ω̇ = τ − ω×(I·ω)` with quaternion
  kinematics `q̇ = ½ [ω, 0] ⊗ q` (JPL convention).
- **Integration:** fixed-step RK4

## Control
- Decoupled LQR for translation and attitude, designed via continuous-time
  algebraic Riccati equation
- The (continuous full-state) stability margins would weaken under discrete sampling
  and the LQR+EKF combination. Robustness is therefore verified via Monte Carlo.

## Navigation
- **Multiplicative EKF** with a 12-dim error state, avoiding the singular
  covariance a 4-component quaternion would produce. Attitude corrections are
  injected multiplicatively; the Joseph-form covariance update is used.
- **Gyro as a predict input.** Attitude is propagated with the bias-corrected
  gyro (ω_hat = ω_m − b_hat) instead of Euler's equation, so the filter needs no
  inertia or torque info. The gyro has no innovation; its bias is an estimated state.
- **Error-state Jacobian** is analytic: discretized as F_d ≈ I + F_c·dt
- **Sequential updates:** star tracker (H = [0 0 I 0]), then lidar (nonlinear
  range/azimuth/elevation model; its Jacobian couples position and attitude).
- **Process noise** is applied in error-state space: gyro ARW and bias random
  walk from the sensor config, plus a deliberately small translational margin
  (scale 1e-3, not a fitted value) for unmodeled accelerations.
- Timing: the first call skips predict; every later call predicts t_{k−1}→t_k
  and then fuses the measurement taken at t_k.

Results and discussion are in `results/summary.md`.

## Limitations
- Sensor noise is white plus gyro bias random walk; no star-tracker systematic
  bias, lidar bias, or misalignment. The lidar angle noise and gyro bias random
  walk are assumed.
- Translation has no unmodeled disturbances (no J2, thruster errors), so the
  translational process noise is a margin, and dropout coasting is optimistic.
- Actuators are saturation-only