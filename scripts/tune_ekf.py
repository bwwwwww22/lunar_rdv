"""EKF tuning study: sweep the process-noise Q scale factor and characterize
the resulting steady-state estimation error. 
Tests how trusting the physical dynamics model more (small Q) versus 
trusting the sensor measurements more (large Q) impacts estimation accuracy.
Produces a tuning curve demonstrating deliberate selection of Q"""

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from scripts.run_openloop import build_params
from src.utils.quaternions import quat_error, quat_multiply, quat_normalize
from src.analysis.monte_carlo import (default_sensor_config, default_R,
        make_sensor_suite, default_ekf_config)

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

def run_one(params, q_trans_scale, seed=42, dt=1.0, t_end=3000.0):
    """Run one closed-loop docking with the position/velocity blocks of Q scaled by
    q_trans_scale (the gyro blocks stay at their matched values).
    Returns steady-state RMS estimation error for pos, vel, att, gyro bias."""
    # LQR (fixed at previous tuning)
    Q_trans = np.diag([1.0]*3 + [0.1]*3)
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0]*3 + [1.0]*3)
    R_att   = np.eye(3) * 100.0

    # truth sensor noise and the matching EKF R/P0/Q (all from one config)
    cfg = default_sensor_config()
    R_ekf = default_R(cfg)
    P0, base_Q_ekf = default_ekf_config(cfg, q_trans_scale=1.0)   # unscaled baseline to sweep

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)

    x0_est = x0.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] = 0.0   # filter state 10:13 holds the gyro-bias estimate b_hat, not w
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(dq, x0_est[6:10]))

    sensor = make_sensor_suite(dt, seed, cfg)
    # scale only the translation blocks: they hold unmodeled-acceleration margin that this
    # sim doesn't need, while the attitude/bias blocks come from the gyro spec and get worse if shrunk
    Q_ekf = base_Q_ekf.copy()
    Q_ekf[0:6, 0:6] *= q_trans_scale
    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)

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
    # gyro-bias error (rad/s); the old omega-error is now just gyro noise, so it can't be tuned
    bias_rms = np.sqrt(np.mean(np.sum((log["b_hat"][ss] - log["b_true"][ss])**2, axis=1)))

    att_errs = np.array([
        np.linalg.norm(quat_error(est[k, 6:10], truth[k, 6:10]))
        for k in range(n)
    ])
    att_rms = np.sqrt(np.mean(att_errs[ss]**2))

    return pos_rms, vel_rms, att_rms, bias_rms


def main():
    params = build_params()
    q_scales = [1e-6, 1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # translation-block scale

    results = []
    for q in q_scales:
        p, v, a, r = run_one(params, q)
        results.append((q, p, v, a, r))
        print(f"q_trans_scale={q:8.4f} | pos_rms={p:.3f} m | vel_rms={v:.4f} m/s | "
              f"att_rms={np.rad2deg(a):.3f} deg | bias_rms={np.rad2deg(r)*3600:.4f} deg/hr")

    results = np.array(results)

    fig, axs = plt.subplots(2, 2, figsize=(6,6), tight_layout=True)
    axs = axs.ravel()
    axs[0].loglog(results[:,0], results[:,1], "o-"); axs[0].set_ylabel("Position RMS [m]")
    axs[1].loglog(results[:,0], results[:,2], "o-"); axs[1].set_ylabel("Velocity RMS [m/s]")
    axs[2].loglog(results[:,0], np.rad2deg(results[:,3]), "o-"); axs[2].set_ylabel("Attitude RMS [deg]")
    axs[3].loglog(results[:,0], np.rad2deg(results[:,4])*3600, "o-");
    axs[3].set_ylabel("Gyro bias RMS [deg/hr]")
    for ax in axs:
        ax.set_xlabel("translation Q scale"); 
        ax.set_box_aspect(1)
    fig.suptitle("EKF tuning: steady-state error vs translation process-noise scale")
    plt.savefig("results/ekf_tuning_curve.png", dpi=300)


if __name__ == "__main__":
    main()