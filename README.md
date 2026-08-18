# 6-DOF SIL Simulator for Lunar Rendezvous & Docking
A closed-loop 6-DOF SIL Python simulator for spacecraft proximity
operations and docking in lunar orbit. Modular architecture isolates the
vehicle physics plant from the GN&C flight software. Core focus: a
Multiplicative Extended Kalman Filter (MEKF) for relative navigation and an
LQR flight controller, validated via consistency testing and Monte Carlo
robustness analysis.

## Status
- [x] 6-DOF plant (Clohessy–Wiltshire translation + quaternion attitude)
- [x] LQR flight control (decoupled translation/attitude)
- [x] Synthetic pose sensor
- [x] LQR tuning (convergence vs. propellant tradeoff)
- [x] MEKF navigation (w/ assumptions)
- [x] MEKF tuning + NEES/NIS consistency validation
- [x] Monte Carlo robustness + 2D recoverability envelope
- [x] Off-nominal sensor scenario (measurement dropout)

## Architecture

A SIL boundary is enforced between the environment (truth) and the flight
software (which sees only noisy sensor data):

    PLANT (truth) → SENSOR (adds noise) → NAV (EKF estimate) → GUIDANCE (setpoint)
      → CONTROL (LQR) → ACTUATORS (saturation) → PLANT (next step)

- **Environment / plant** (`src/environment/`): 6-DOF physics. Never sees
  the controller's intent, only the post-saturation applied control.
- **Flight software** (`src/fsw/`): sensors, navigation, guidance, control.
  Never accesses the true state, only noisy measurements.
- **Sim** (`src/sim/`): fixed-rate scheduler and closed-loop runner
- **Analysis** (`src/analysis/`): consistency (NEES/NIS) and Monte Carlo

## Conventions
- **Frames:** translation in Hill/LVLH frame (x=radial, y=along-track,
  z=cross-track); rotation in body frame.
- **Quaternions:** scalar-first `[w, x, y, z]`, Hamilton convention,
  body-relative-to-reference (not JPL).
- **Units:** meters, seconds, radians, kilograms

## State vector (13) and error state (12)
The full state carries a 4-component quaternion; the EKF operates on a
12-dimensional error state using a 3-parameter small-angle attitude error.

| Index | Symbol | Meaning | Frame | Units |
|-------|--------|---------|-------|-------|
| 0:3   | r      | relative position | Hill | m |
| 3:6   | v      | relative velocity | Hill | m/s |
| 6:10  | q      | attitude quaternion (body→ref) | — | — |
| 10:13 | ω      | angular velocity | body | rad/s |

Error state (12): `[δr(3), δv(3), δθ(3), δω(3)]`, where δθ is a small-angle
attitude error. Corrections are injected multiplicatively into the quaternion.

## Dynamics
- **Translation:** Clohessy–Wiltshire equations, lunar mean motion
  `n = √(µ_moon / a³)` for a 100 km circular low lunar orbit. No atmospheric drag.
- **Rotation:** rigid-body Euler equation `I·ω̇ = τ − ω×(I·ω)` with quaternion
  kinematics `q̇ = ½ q ⊗ [0, ω]`.
- **Integration:** fixed-step RK4

## Control
- Decoupled LQR for translation and attitude, designed via continuous-time
  algebraic Riccati equation
- Stability margins: the classical guaranteed margins (≥6 dB gain, ≥60° phase)
  weaken under discrete sampling and EKF-based estimation, so robustness is
  verified via Monte Carlo.

## Navigation
- **Multiplicative EKF** with a 12-dim error state, avoiding the singular
  covariance a 4-component quaternion would produce. Attitude corrections are
  injected multiplicatively; the Joseph-form covariance update is used.
- **Error-state Jacobian** uses central differences on the full state, then
  maps quaternion perturbations to small-angle perturbations
- **Process noise** is applied in error-state space, weighted primarily on the
  velocity and angular-rate channels (representing unmodeled accelerations and
  torques); position/attitude blocks inherit uncertainty through integration.