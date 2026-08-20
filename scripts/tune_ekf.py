"""EKF tuning study: sweep the process-noise Q scale factor and characterize
the resulting steady-state estimation error. 
Tests how trusting the physical dynamics model more (small Q) versus 
trusting the sensor measurements more (large Q) impacts estimation accuracy.
Produces a tuning curve demonstrating deliberate selection of Q"""

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error
from scripts.run_nominal import build_params
from src.utils.quaternions import quat_multiply, quat_normalize

def base_P0():
    return np.diag([
        1.0, 1.0, 1.0,
        0.1, 0.1, 0.1,
        np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,
        np.deg2rad(0.1)**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2,
    ])


def base_Q():
    """Baseline process-noise covariance (12x12). Scaled by q_scale below."""
    return np.diag([
        1e-4, 1e-4, 1e-4,
        1e-3, 1e-3, 1e-3,
        1e-8, 1e-8, 1e-8,
        1e-6, 1e-6, 1e-6,
    ])


def sensor_R(pos_std, vel_std, att_std, rate_std):
    return np.diag(
        [pos_std**2]*3 + [vel_std**2]*3 +
        [att_std**2]*3 + [rate_std**2]*3
    )


def run_one(params, q_scale, seed=42, dt=1.0, t_end=3000.0):
    """Run one closed-loop docking with Q scaled by q_scale.
    Returns steady-state RMS estimation error for pos, vel, att, rate."""
    # LQR (fixed at previous tuning)
    Q_trans = np.diag([1.0]*3 + [0.1]*3)
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0]*3 + [1.0]*3)
    R_att   = np.eye(3) * 100.0

    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.9962, 0.0872, 0.0, 0.0]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_std, vel_std, att_std, rate_std, seed=seed)

    x0_est = x0.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] += np.deg2rad(np.array([0.1, -0.1, 0.1]))
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(x0_est[6:10], dq))

    # pass different q_scale values for Q into the EKF initialization
    nav = EKFNav(params, x0_est, base_P0(), base_Q() * q_scale,
                 sensor_R(pos_std, vel_std, att_std, rate_std))
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)

    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators,
                    scheduler)
    log = sim.run(t_end=t_end)

    # Steady-state RMS errors (last 40% of run, past transients)
    truth, est = log["x_truth"], log["x_est"]
    n = len(log["t"])
    ss = slice(int(0.6 * n), n)

    pos_rms = np.sqrt(np.mean(np.sum((est[ss, 0:3] - truth[ss, 0:3])**2, axis=1)))
    vel_rms = np.sqrt(np.mean(np.sum((est[ss, 3:6] - truth[ss, 3:6])**2, axis=1)))
    rate_rms = np.sqrt(np.mean(np.sum((est[ss, 10:13] - truth[ss, 10:13])**2, axis=1)))

    att_errs = np.array([
        np.linalg.norm(quat_error(est[k, 6:10], truth[k, 6:10]))
        for k in range(n)
    ])
    att_rms = np.sqrt(np.mean(att_errs[ss]**2))

    return pos_rms, vel_rms, att_rms, rate_rms


def main():
    params = build_params()
    q_scales = [0.01, 0.1, 1.0, 10.0, 100.0]

    results = []
    for q in q_scales:
        p, v, a, r = run_one(params, q)
        results.append((q, p, v, a, r))
        print(f"q_scale={q:7.2f} | pos_rms={p:.3f} m | vel_rms={v:.4f} m/s | "
              f"att_rms={np.rad2deg(a):.3f} deg | rate_rms={np.rad2deg(r):.4f} deg/s")

    results = np.array(results)

    fig, axs = plt.subplots(2, 2, figsize=(6,6), tight_layout=True)
    axs = axs.ravel()
    axs[0].loglog(results[:,0], results[:,1], "o-"); axs[0].set_ylabel("Position RMS [m]")
    axs[1].loglog(results[:,0], results[:,2], "o-"); axs[1].set_ylabel("Velocity RMS [m/s]")
    axs[2].loglog(results[:,0], np.rad2deg(results[:,3]), "o-"); axs[2].set_ylabel("Attitude RMS [deg]")
    axs[3].loglog(results[:,0], np.rad2deg(results[:,4]), "o-"); axs[3].set_ylabel("Rate RMS [deg/s]")
    for ax in axs:
        ax.set_xlabel("Q scale factor"); 
        ax.set_box_aspect(1)
    fig.suptitle("EKF tuning: steady-state error vs process noise scale")
    plt.savefig("results/ekf_tuning_curve.png", dpi=300)


if __name__ == "__main__":
    main()