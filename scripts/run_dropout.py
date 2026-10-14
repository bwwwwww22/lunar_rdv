"""Off-nominal scenario: sensor dropout during docking. 

During the dropout window the MEKF runs predict-only (no measurement update);
covariance grows; translation coasts on CW + cmd force and attitude on the
bias-corrected gyro (star tracker & lidar dropout, the gyro stays on). """

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import DropoutSensor
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error
from scripts.run_openloop import build_params
from src.utils.quaternions import quat_multiply, quat_normalize
from src.analysis.monte_carlo import (default_sensor_config, default_R,
        make_sensor_suite, default_ekf_config)

def run_with_dropout(dropout_windows, drop=("star", "lidar"), t_end=3000.0, dt=1.0, seed=42):
    params = build_params()

    Q_trans = np.diag([1.0]*3 + [0.1]*3)
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0]*3 + [1.0]*3)
    R_att   = np.eye(3) * 100.0

    # truth sensor noise and the matching EKF R/P0/Q (all from one config)
    cfg = default_sensor_config()
    R_ekf = default_R(cfg)
    P0, Q_ekf = default_ekf_config(cfg)

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)

    sensor = make_sensor_suite(dt, seed, cfg)
    # Drop only star tracker & lidar. Gyro keeps running because it
    # drives predict, w/o it there'd be nothing to propagate attitude with
    ##sensor.star = DropoutSensor(sensor.star, dropout_windows, dt=dt)
    ##sensor.lidar = DropoutSensor(sensor.lidar, dropout_windows, dt=dt)
    if isinstance(drop, str):
        drop = (drop,)
    for name in drop:
        setattr(sensor, name, DropoutSensor(getattr(sensor, name), dropout_windows, dt=dt))

    x0_est = x0.copy()
    #x0_est[0:3] += [0.5, -0.5, 0.3]
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

    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)

    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)

    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators,
                    scheduler)
    log = sim.run(t_end=t_end)
    return log, dock_position


def plot_comparison(results_list, figure_title, figure_name):
    """
    Plots estimation errors for multiple runs within a single figure using 
    a grid of subplots (N_runs x 2 subplots).
    Each row corresponds to a single scenario/run.
    """
    n_runs = len(results_list)
    
    # Create a grid: n_runs rows, 2 columns (pos err, att err)
    fig, axs = plt.subplots(
        2,  n_runs,
        figsize=(3 * n_runs, 6), 
        sharex=True, 
        tight_layout=True
    )

    # Ensure axs is 2D even if there's only 1 run
    if n_runs == 1: axs = np.expand_dims(axs, axis=0)

    fig.suptitle(figure_title)

    for i, res in enumerate(results_list):
        log = res["log"]
        label = res["label"]
        windows = res["dropout_windows"]

        ## old: the runner never fills x_est[0] (zeros), so skip the first sample
        s = slice(0,None) #1, None)
        t = log["t"][s]
        truth, est = log["x_truth"][s], log["x_est"][s]

        pos_err = np.linalg.norm(est[:, 0:3] - truth[:, 0:3], axis=1)
        att_err = np.rad2deg(np.array([
            np.linalg.norm(quat_error(est[k, 6:10], truth[k, 6:10]))
            for k in range(len(t))
        ]))

        ax_pos = axs[0,i]; ax_att = axs[1,i]
        ax_pos.plot(t, pos_err, color="tab:blue", linewidth=1.2)
        ax_att.plot(t, att_err, color="tab:orange", linewidth=1.2)

        # Highlight dropout windows
        for (t0, t1) in windows:
            ax_pos.axvspan(t0, t1, color="red", alpha=0.15)
            ax_att.axvspan(t0, t1, color="red", alpha=0.15)

        ax_pos.set_ylabel("Pos Err [m]"); ax_att.set_ylabel("Att Err [deg]")
        ax_pos.set_title(f"{label}", fontsize=10)

    for ax in axs[-1]: ax.set_xlabel("Time [s]")
    for ax in axs.flat: ax.set_box_aspect(1.0)

    plt.savefig(figure_name, dpi=300, bbox_inches="tight")

def main():
    # ---Scenario 1: Varying Durations from a Fixed Start Time---
    print("\n--- Running Varying Duration Sweep ---")
    start_time = 30 #1000.0
    durations = [300, 1000, 2000, 2500, 3000]
    duration_results = []

    for duration in durations:
        dropout_windows = [(start_time, start_time + duration)]
        log, dock = run_with_dropout(dropout_windows, drop=("lidar"))
        
        # Calculate summary metrics
        truth = log["x_truth"]
        fpe = np.linalg.norm(truth[-1, 0:3] - dock)
        fae = np.rad2deg(np.linalg.norm(quat_error(truth[-1, 6:10], np.array([0, 0, 0, 1]))))
        success = fpe < 1.0 and fae < 5.0

        print(f"Dropout {duration:4d}s -> Final Pos Err: {fpe:.3f}m | "
              f"Att Err: {fae:.3f}deg | Success: {success}")

        duration_results.append({
            "label": f"Duration {duration}s",
            "log": log,
            "dropout_windows": dropout_windows,
            "dock_position": dock
        })
    plot_comparison(duration_results, 
                    figure_title="EKF Estimation Error Across Various Dropout Durations",
                    figure_name="results/dropout_durations_dropslidar.png")


    #---Scenario 2: Varying Start Times---
    print("\n--- Running Varying Start Times---")
    #combo_windows = [(500, 1500), (500, 2000), (1000, 1500), (1000, 2000)]
    dur = 2000
    combo_windows = [(s, dur) for s in (500, 750, 900, 950, 1000)]
    combo_results = []

    for start, duration in combo_windows:
        dropout_windows = [(float(start), float(start + duration))]
        log, dock = run_with_dropout(dropout_windows, drop=("lidar"))

        truth = log["x_truth"]
        fpe = np.linalg.norm(truth[-1, 0:3] - dock)
        fae = np.rad2deg(np.linalg.norm(quat_error(truth[-1, 6:10], np.array([0, 0, 0, 1]))))
        success = fpe < 1.0 and fae < 5.0

        print(f"Window [{start}:{start+duration}] -> Final Pos Err: {fpe:.3f}m | "
              f"Att Err: {fae:.3f}deg | Success: {success}")

        combo_results.append({
            "label": f"Start {start}s (Dur {duration}s)",
            "log": log,
            "dropout_windows": dropout_windows,
            "dock_position": dock
        })

    plot_comparison(combo_results, 
                    figure_title="EKF Estimation Error Across Various Dropout Start Times", 
                    figure_name="results/dropout_start_droplidar.png")


if __name__ == "__main__":
    main()