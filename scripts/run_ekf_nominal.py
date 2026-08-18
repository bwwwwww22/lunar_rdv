"""Nominal lunar docking run"""

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
from scripts.run_openloop import build_params

def main():
    params = build_params()

    # --- LQR tuning matrices
    # arbitrary values but this Q setting (penalizes state error) has position weighted 
    # 10× more than velocity, meaningwe care more about being at the dock than about Rthe exact speed
    Q_trans = np.diag([1.0, 1.0, 1.0,    # position error (x,y,z)
                       0.1, 0.1, 0.1])   # velocity error (vx,vy,vz)
    R_trans = np.diag([1.0, 1.0, 1.0]) * 10.0    # thrust cost; bigger = stingier with fuel
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.diag([1.0, 1.0, 1.0]) * 100.0   # make using torque very expensive

    # --- EKF tuning, [δr(3), δv(3), δθ(3), δω(3)]
    # sort of arbitrary, but the idea is that we have a good handle on the translational state
    # (position & velocity) and worse on the attitude (small-angle error) and angular rate
    # The process noise is set to be small for position and attitude, 
    # and larger for velocity and angular rate, which are the primary channels of uncertainty.
    P0 = np.diag([
        1.0, 1.0, 1.0,               # initial pos uncertainty (m^2)
        0.1, 0.1, 0.1,               # velocity  (m/s)^2
        np.deg2rad(5)**2,            # attitude (rad^2), ~5 deg 1-sigma
        np.deg2rad(5)**2,
        np.deg2rad(5)**2,
        np.deg2rad(0.1)**2,          # angular rate (rad/s)^2
        np.deg2rad(0.1)**2,
        np.deg2rad(0.1)**2,
    ])
    Q_ekf = np.diag([
        1e-4, 1e-4, 1e-4,            # pos process noise
        1e-3, 1e-3, 1e-3,            # velocity
        1e-6, 1e-6, 1e-6,            # attitude (small)
        1e-6, 1e-6, 1e-6,            # angular rate
    ])
    # R matches the sensor noise set below
    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)
    R_ekf = np.diag([
        pos_std**2]*3 + [vel_std**2]*3 +
        [att_std**2]*3 + [rate_std**2]*3
    )
    
    # --- Initial condition: chaser offset from dock, small attitude error
    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]   # position (Hill); start 100 m behind, offset radial/cross
    #x0[3:6] = [0.0, 0.0, 0.0]        # velocity (Hill)
    x0[6:10] = [0.9962, 0.0872, 0.0, 0.0]  # quaternion (attitude body→ref, scalar-first); here ~10 deg roll error
    x0[10:13] = [0.0, 0.0, 0.0]      # body-frame angular rate

    # Docking target setpoint (Hill)
    dock_position = np.array([0.0, -5.0, 0.0])  # dock 5 m along-track behind (same radial altitude & cross-track plane as target)
    
    # --- EKF initial estimate: a further small offset from the dispersed truth
    # simulates real-life uncertainty at initialization: the EKF does not 
    # know the exact true position of the spacecraft right at t=0; 
    # it starts with a small initial estimate error (up to 0.5 m) and relies on
    # its measurement updates to converge
    x0_est = x0.copy()
    x0_est[0:3] += np.array([0.5, -0.5, 0.3])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_noise_std=pos_std, vel_noise_std=vel_std,
                        att_noise_std=att_std,
                        rate_noise_std=rate_std, seed=42)
    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=1.0)

    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators, scheduler)

    # Run for a few thousand seconds (dynamics are slow at lunar orbit rate)
    log = sim.run(t_end=3000.0)

    plot_ekf_performance(log)
    print_error_stats(log)


def plot_ekf_performance(log):
    t = log["t"]
    truth = log["x_truth"]
    est = log["x_est"]

    # --- Estimation error over time ---
    pos_err = np.linalg.norm(est[:, 0:3] - truth[:, 0:3], axis=1)
    vel_err = np.linalg.norm(est[:, 3:6] - truth[:, 3:6], axis=1)
    rate_err = np.linalg.norm(est[:, 10:13] - truth[:, 10:13], axis=1)

    # Attitude error via quaternion residual (small-angle vector norm
    from src.utils.quaternions import quat_error
    att_err = np.array([
        np.linalg.norm(quat_error(est[k, 6:10], truth[k, 6:10]))
        for k in range(len(t))
    ])

    fig, axs = plt.subplots(5, 1, figsize=(9, 9), sharex=True, tight_layout=True)
    axs[0].plot(t, pos_err);              axs[0].set_ylabel("‖pos err‖ [m]")
    axs[1].plot(t, vel_err);              axs[1].set_ylabel("‖vel err‖ [m/s]")
    axs[2].plot(t, np.rad2deg(att_err));  axs[2].set_ylabel("‖att err‖ [deg]")
    axs[3].plot(t, np.rad2deg(rate_err)); axs[3].set_ylabel("‖ω err‖ [deg/s]")
    for ax in axs[:4]: ax.set_xlabel("time [s]")
    for ax in axs: ax.set_xlim([0, 3000])
    fig.suptitle("EKF estimation error vs truth")

    # --- Docking convergence (same as Week 1 plot, sanity check) ---
    ax = axs[4]
    ax.plot(t, truth[:, 0], label="radial x")
    ax.plot(t, truth[:, 1], label="along-track y")
    ax.plot(t, truth[:, 2], label="cross-track z")
    ax.axhline(0, color="k", ls='--', zorder=0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("position [m]")
    ax.set_title("Docking (truth)"); ax.legend(ncols=3); #ax.grid(True)

    plt.savefig("results/ekf_nominal.png", dpi=300, bbox_inches="tight")
    #plt.show()

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
    err_rate = est[ss, 10:13] - truth[ss, 10:13]
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
    report("Rate [deg/s]",    err_rate, unit_scale=180/np.pi, unit="deg/s")


if __name__ == "__main__":
    main()
