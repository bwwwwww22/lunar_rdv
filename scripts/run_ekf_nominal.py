"""Nominal lunar docking run"""

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

def main():
    params = build_params()
        
    # LQR (fixed)
    Q_trans = np.diag([1.0, 1.0, 1.0, 
                        0.1, 0.1, 0.1])
    R_trans = np.diag([1.0, 1.0, 1.0]) * 10.0 
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.diag([1.0, 1.0, 1.0]) * 100.0

    # truth sensor noise and the matching EKF R/P0/Q (all from one config)
    cfg = default_sensor_config()
    R_ekf = default_R(cfg)
    P0, Q_ekf = default_ekf_config(cfg)
    
    # --- Initial condition: chaser offset from dock, small attitude error
    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]   # position (Hill); start 100 m behind, offset radial/cross
    #x0[3:6] = [0.0, 0.0, 0.0]        # velocity (Hill)
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]  # quaternion (attitude ref→body, JPL scalar-last); here ~10 deg roll error
    x0[10:13] = [0.0, 0.0, 0.0]      # body-frame angular rate

    # Docking target setpoint (Hill)
    dock_position = np.array([0.0, -5.0, 0.0])  # dock 5 m along-track behind (same radial altitude & cross-track plane as target)
    
    # --- EKF initial estimate: a further small offset from the dispersed truth
    # simulates real-life uncertainty at initialization: the EKF does not 
    # know the exact true position of the spacecraft right at t=0; 
    # it starts with a small initial estimate error (up to 0.5 m) and relies on
    # its measurement updates to converge
    x0_est = x0.copy()

    # Position: 1.0 m per axis (matches P0 = 1.0 m²)
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    # Velocity: 0.316 m/s per axis (matches P0 = 0.1 (m/s)²)
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] = 0.0   # filter state 10:13 hold the b_hat (gyro bias est.), not w

    # Attitude: 5 deg per axis (matches P0 = deg2rad(5)²)
    # Apply MULTIPLICATIVELY
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(dq, x0_est[6:10]))
    
    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    seed = 42
    dt = 1.0
    sensor = make_sensor_suite(dt, seed, cfg)
    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)

    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators, scheduler)

    # Run for a few thousand seconds (dynamics are slow at lunar orbit rate)
    log = sim.run(t_end=3000.0)

    plot_ekf_performance(log)
    print_error_stats(log)


def plot_ekf_performance(log):
    # old:the runner never fills est[0] (zeros), so drop the first sample;
    # otherwise t=0 shows a weird, large "error"
    s = slice(0,None) #1, None)
    t = log["t"][s]
    truth = log["x_truth"][s]
    est = log["x_est"][s]
    b_hat, b_true = log["b_hat"][s], log["b_true"][s]

    # Estimation error over time
    pos_err = np.linalg.norm(est[:, 0:3] - truth[:, 0:3], axis=1)
    vel_err = np.linalg.norm(est[:, 3:6] - truth[:, 3:6], axis=1)

    # Attitude error via quaternion residual (small-angle vector norm
    att_err = np.array([
        np.linalg.norm(quat_error(est[k, 6:10], truth[k, 6:10]))
        for k in range(len(t))
    ])

    fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=True, tight_layout=True)
    axs[0].semilogy(t, pos_err)
    axs[0].set_ylabel("‖pos err‖ [m]")             
    axs[1].semilogy(t, vel_err)
    axs[1].set_ylabel("‖vel err‖ [m/s]")
    axs[2].semilogy(t, np.rad2deg(att_err))
    axs[2].set_ylabel("‖att err‖ [deg]")

    # gyro bias: est vs truth (dashed) per axis
    # This replaces the old omega-error panel: w_hat = w_m - b_hat
    # is unfiltered, so its error is just gyro noise
    to_deg_hr = 180 / np.pi * 3600
    for i, axis in enumerate("xyz"):
        axs[3].plot(t, b_true[:, i] * to_deg_hr, "--", color=f"C{i}")
        axs[3].plot(t, b_hat[:, i] * to_deg_hr, color=f"C{i}", label=f"b_{axis}")
    axs[3].set_ylabel("gyro bias [deg/hr]")
    axs[3].legend(ncols=3, title="solid = estimate, dashed = truth")
    for ax in axs: ax.set_xlim([0, 3000])
    fig.suptitle("EKF estimation error vs truth")

    # Docking convergence
    ax = axs[4]
    ax.plot(t, truth[:, 0], label="radial x")
    ax.plot(t, truth[:, 1], label="along-track y")
    ax.plot(t, truth[:, 2], label="cross-track z")
    ax.axhline(0, color="k", ls='--', zorder=0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("position [m]")
    ax.set_title("Docking (truth)"); ax.legend(ncols=3); #ax.grid(True)

    plt.savefig("results/ekf_nominal.png", dpi=300, bbox_inches="tight")
    #plt.show()

    # dignose the 'bump' in pos/vel error
    unshifted = np.linalg.norm(log["x_est"][1:, 0:3] - log["x_truth"][1:, 0:3], axis=1)
    shifted = np.linalg.norm(log["x_est"][1:, 0:3] - log["x_truth"][:-1, 0:3], axis=1)
    w = slice(30, 90)       # around the ~50 s bump
    print(unshifted[w].max(), shifted[w].max())
    print(shifted[:200].argmax())   # where does the 0.427 come from?


def print_error_stats(log):
    """Print signed estimation error statistics per channel. Bias (mean)
    should be near zero; std should match sensor-noise-level after filtering."""
    from src.utils.quaternions import quat_error
    truth, est = log["x_truth"], log["x_est"]
    n = len(log["t"])
    ss = slice(int(0.6 * n), n)   # define steady state as last 40% of run

    # Signed errors per channel
    err_pos  = est[ss, 0:3]   - truth[ss, 0:3]
    err_vel  = est[ss, 3:6]   - truth[ss, 3:6]
    err_bias = log["b_hat"][ss] - log["b_true"][ss]   # b_hat - b (same sign convention as pos/vel/att above)
    err_att  = np.array([quat_error(est[k, 6:10], truth[k, 6:10]) for k in range(len(truth))])[ss]

    def report(name, err, unit_scale=1.0, unit=""):
        mean = err.mean(axis=0) * unit_scale
        std  = err.std(axis=0)  * unit_scale
        print(f"{name}:")
        print(f"  mean (bias): {mean}  {unit}")
        print(f"  std        : {std}   {unit}")
        print(f"  |mean|/std : {np.abs(mean)/std}  (should be <~ 0.3)")

    print("\n=== Steady-state signed estimation errors ===")
    report("Position [m]",    err_pos)
    report("Velocity [m/s]",  err_vel)
    report("Attitude [deg]",  err_att,  unit_scale=180/np.pi, unit="deg")
    report("Gyro bias [deg/hr]", err_bias, unit_scale=180/np.pi*3600, unit="deg/hr")


if __name__ == "__main__":
    main()
