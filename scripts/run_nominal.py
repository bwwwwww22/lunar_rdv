"""Nominal lunar docking run"""

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor
from src.fsw.navigation import PassthroughNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_to_euler, quat_error
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

    # --- Initial condition: chaser offset from dock, small attitude error
    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]   # position (Hill); start 100 m behind, offset radial/cross
    #x0[3:6] = [0.0, 0.0, 0.0]        # velocity (Hill)
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]  # quaternion (attitude ref→body, JPL scalar-last); here ~10 deg roll error
    x0[10:13] = [0.0, 0.0, 0.0]      # body-frame angular rate

    # Docking target setpoint (Hill)
    dock_position = np.array([0.0, -5.0, 0.0])  # dock 5 m along-track behind (same radial altitude & cross-track plane as target)

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_noise_std=0.1, vel_noise_std=0.01,
                        att_noise_std=np.deg2rad(0.5),
                        rate_noise_std=np.deg2rad(0.05), seed=42)
    nav = PassthroughNav()
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=1.0)

    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators, scheduler)

    # Run for a few thousand seconds (dynamics are slow at lunar orbit rate)
    log = sim.run(t_end=3000.0)

    final_truth = log["x_truth"][-1]        # last timestep truth state
    dock_attitude = np.array([0.0, 0.0, 0.0, 1.0])

    # Position error
    pos_err = final_truth[0:3] - dock_position
    print("Final position:        ", final_truth[0:3], "m")
    print("Dock setpoint:         ", dock_position, "m")
    print("Final position error:  ", pos_err, "m")
    print("Position error norm:   ", np.linalg.norm(pos_err), "m")

    # Velocity error (should be ~0 at dock)
    print("Final velocity:        ", final_truth[3:6], "m/s")
    print("Velocity error norm:   ", np.linalg.norm(final_truth[3:6]), "m/s")

    # Attitude error (small-angle vector, then convert to degrees)
    att_err_vec = quat_error(final_truth[6:10], dock_attitude)   # rad, 3-vector
    att_err_deg = np.rad2deg(np.linalg.norm(att_err_vec))
    print("Final attitude error:  ", att_err_deg, "deg")

    # Angular rate (should be ~0 at dock)
    print("Final angular rate:    ", np.rad2deg(final_truth[10:13]), "deg/s")

    plot_results(log, dock_position)


def plot_results(log, dock_position):
    t = log["t"]
    xt = log["x_truth"]    # x_truth is the real state of the spacecraft (position, velocity, attitude, rate)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Create figure
    fig = plt.figure(figsize=(9,6), tight_layout=True)
    gs = gridspec.GridSpec(6, 2, figure=fig)

    # left column: relative position
    labels = ["radial x", "along-track y", "cross-track z"]
    ax_pos = [
        fig.add_subplot(gs[0:2, 0]),
        fig.add_subplot(gs[2:4, 0]),
        fig.add_subplot(gs[4:6, 0]),
    ]

    for i in range(3):
        ax_pos[i].plot(t, xt[:, i], label="truth")
        ax_pos[i].axhline(dock_position[i], ls="--", color="k", label="dock")
        ax_pos[i].set_ylabel(f"{labels[i]} [m]")
        ax_pos[i].set_xlabel("time [s]")
        ax_pos[i].legend()
    ax_pos[0].set_title("Relative position → docking setpoint")

    # right column: attitude & control effort
    ax_att = fig.add_subplot(gs[0:3, 1])
    ax_ctrl = fig.add_subplot(gs[3:6, 1])

    # Attitude Error (Euler angles vs time)
    eul = np.array([quat_to_euler(q) for q in xt[:, 6:10]])
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        ax_att.plot(t, np.rad2deg(eul[:, i]), label=name)
    ax_att.axhline(0, ls="--", color="k")
    ax_att.set_xlabel("time [s]")
    ax_att.set_ylabel("attitude [deg]")
    ax_att.set_title("Attitude → target")
    ax_att.legend()
    ax_att.set_box_aspect(1.0)

    # Control Effort
    for i, name in enumerate(["Fx", "Fy", "Fz"]):
        ax_ctrl.plot(t, log["u_applied"][:, i], label=name)
    ax_ctrl.axhline(0, ls="--", color="k")
    ax_ctrl.set_xlabel("time [s]")
    ax_ctrl.set_ylabel("force [N]")
    ax_ctrl.set_title("Applied thrust")
    ax_ctrl.legend()
    ax_ctrl.set_box_aspect(1.0)

    plt.savefig("results/nominal_docking_run.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
