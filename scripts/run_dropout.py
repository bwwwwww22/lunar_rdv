"""Off-nominal scenario: sensor dropout during docking. 

During the dropout window the MEKF runs predict-only (no measurement update);
covariance grows and the estimate coasts on the dynamics model. Demonstrates 
recovery rather than divergence."""

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor, DropoutSensor
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error
from src.analysis.monte_carlo import default_ekf_config
from scripts.run_nominal import build_params


def run_with_dropout(dropout_windows, t_end=3000.0, dt=1.0, seed=42):
    params = build_params()

    Q_trans = np.diag([1.0]*3 + [0.1]*3)
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0]*3 + [1.0]*3)
    R_att   = np.eye(3) * 100.0

    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)
    R_ekf = np.diag([pos_std**2]*3 + [vel_std**2]*3 +
                    [att_std**2]*3 + [rate_std**2]*3)
    P0, Q_ekf = default_ekf_config()

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.9962, 0.0872, 0.0, 0.0]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    base_sensor = PoseSensor(pos_std, vel_std, att_std, rate_std, seed=seed)
    sensor = DropoutSensor(base_sensor, dropout_windows, dt=dt)

    x0_est = x0.copy()
    #x0_est[0:3] += [0.5, -0.5, 0.3]
    # Position: 1.0 m per axis (matches P0 = 1.0 m²)
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    # Velocity: 0.316 m/s per axis (matches P0 = 0.1 (m/s)²)
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    # Angular rate: 0.1 deg/s per axis (matches P0 = deg2rad(0.1)²)
    x0_est[10:13] += np.deg2rad(np.array([0.1, -0.1, 0.1]))
    # Attitude: 5 deg per axis (matches P0 = deg2rad(5)²)
    # Apply MULTIPLICATIVELY (not additively — attitude trap)
    from src.utils.quaternions import quat_multiply, quat_normalize
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(x0_est[6:10], dq))
    
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

        t = log["t"]
        truth, est = log["x_truth"], log["x_est"]

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

    #plt.show()
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")

def main():
    # ---Scenario 1: Varying Durations from a Fixed Start Time---
    print("\n--- Running Varying Duration Sweep ---")
    start_time = 1000.0
    durations = [300, 600, 1000, 1500, 2000]
    duration_results = []

    for duration in durations:
        dropout_windows = [(start_time, start_time + duration)]
        log, dock = run_with_dropout(dropout_windows)
        
        # Calculate summary metrics
        truth = log["x_truth"]
        fpe = np.linalg.norm(truth[-1, 0:3] - dock)
        fae = np.rad2deg(np.linalg.norm(quat_error(truth[-1, 6:10], np.array([1, 0, 0, 0]))))
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
                    figure_name="results/dropout_durations.png")


    #---Scenario 2: Varying Start Times & Durations---
    print("\n--- Running Varying Start & Duration Combinations ---")
    combo_windows = [(1000, 1500), (1000, 1800), (500, 1500), (500, 2000)]
    combo_results = []

    for start, duration in combo_windows:
        dropout_windows = [(float(start), float(start + duration))]
        log, dock = run_with_dropout(dropout_windows)

        truth = log["x_truth"]
        fpe = np.linalg.norm(truth[-1, 0:3] - dock)
        fae = np.rad2deg(np.linalg.norm(quat_error(truth[-1, 6:10], np.array([1, 0, 0, 0]))))
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
                    figure_title="EKF Estimation Error Across Start/Duration Combinations", 
                    figure_name="results/dropout_start_duration.png")


if __name__ == "__main__":
    main()