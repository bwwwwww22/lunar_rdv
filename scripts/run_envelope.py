"""Robustness envelope: sweep initial position offset vs initial attitude
error on a grid, run one docking sim per grid point, and map which
initial conditions succeed within the propellant budget."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error, quat_multiply, quat_normalize
from src.analysis.monte_carlo import default_ekf_config
from scripts.run_nominal import build_params

PROPELLANT_BUDGET = 5000.0
DATA_FILE = "../results/envelope_data.npz"

def make_ic(nominal_x0, pos_offset, vel_offset):
    """Place the chaser pos_offset [m] further out along a fixed direction
    commented out: att_offset_deg [deg] attitude error about x-axis."""
    
    x0 = nominal_x0.copy()
    x0[1] -= pos_offset          # along-track position offset [m]
    x0[4] += vel_offset          # along-track velocity offset [m/s]

    # Attitude error about body x-axis
    #dtheta = np.deg2rad(att_offset_deg)
    #dq = quat_normalize(np.array([np.cos(dtheta/2), np.sin(dtheta/2), 0, 0]))
    #x0[6:10] = quat_normalize(quat_multiply(x0[6:10], dq))
    return x0

def run_one(nominal_x0, pos_offset, vel_offset, params, dock_position,
            t_end=3000.0, dt=1.0, seed=42):
    Q_trans = np.diag([1.0]*3 + [0.1]*3)
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0]*3 + [1.0]*3)
    R_att   = np.eye(3) * 100.0

    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)
    R_ekf = np.diag([pos_std**2]*3 + [vel_std**2]*3 +
                    [att_std**2]*3 + [rate_std**2]*3)
    P0, Q_ekf = default_ekf_config()

    x0 = make_ic(nominal_x0, pos_offset, vel_offset)
    x0_est = x0.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] += np.deg2rad(np.array([0.1, -0.1, 0.1]))
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(x0_est[6:10], dq))

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_std, vel_std, att_std, rate_std, seed=seed)
    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)
    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators,
                    scheduler)
    log = sim.run(t_end=t_end)

    truth = log["x_truth"]
    final_pos_err = np.linalg.norm(truth[-1, 0:3] - dock_position)
    final_att_err = np.rad2deg(np.linalg.norm(
        quat_error(truth[-1, 6:10], np.array([1, 0, 0, 0]))))
    propellant = np.sum(np.abs(log["u_applied"][:, 0:3])) * dt

    success = (final_pos_err < 1.0 and final_att_err < 5.0
               and propellant < PROPELLANT_BUDGET)
    return propellant, success


def main():
    # Load cached data if it exists
    if os.path.exists(DATA_FILE):
        print(f"Loading cached simulation data from {DATA_FILE}...")
        data = np.load(DATA_FILE)
        pos_offsets = data['pos_offsets']
        vel_offsets = data['vel_offsets']
        prop_grid = data['prop_grid']
        succ_grid = data['succ_grid']
    else:
        print("No cache found. Run sims...")
        params = build_params()
        nominal_x0 = np.zeros(13)
        nominal_x0[0:3] = [50.0, -100.0, 20.0]
        nominal_x0[6:10] = [1.0, 0.0, 0.0, 0.0]
        dock_position = np.array([0.0, -5.0, 0.0])

        # 16x16 grid of position offset (x) vs velocity offset (y)
        pos_offsets = np.linspace(0, 150, 16)     # along-track position offset
        vel_offsets = np.linspace(0, 5, 16)       # along-track velocity offset

        prop_grid = np.zeros((len(vel_offsets), len(pos_offsets)))
        succ_grid = np.zeros_like(prop_grid, dtype=bool)

        total = len(pos_offsets) * len(vel_offsets)
        count = 0
        for i, vel in enumerate(vel_offsets):        # rows = vel
            for j, pos in enumerate(pos_offsets):    # cols = pos
                prop, succ = run_one(nominal_x0, pos, vel, params, dock_position)
                prop_grid[i, j] = prop
                succ_grid[i, j] = succ
                count += 1
                if count % 16 == 0:
                    print(f"  {count}/{total}")

        # Save data array to disk
        np.savez(DATA_FILE, 
                 pos_offsets=pos_offsets, 
                 vel_offsets=vel_offsets, 
                 prop_grid=prop_grid, 
                 succ_grid=succ_grid)
        print(f"Saved simulation results to {DATA_FILE}")

    plot_envelope(pos_offsets, vel_offsets, prop_grid, succ_grid)

    # summary
    n_total = succ_grid.size
    n_success = succ_grid.sum()
    print("\n=== Envelope summary ===")
    print(f"Grid: {len(pos_offsets)} position x {len(vel_offsets)} velocity "
          f"= {n_total} cells")
    print(f"Overall success fraction: {n_success/n_total*100:.1f}%")
    print(f"Propellant range: {prop_grid.min():.0f} to {prop_grid.max():.0f}")

    # For each velocity row, find the max position offset that still succeeds
    print("\nSuccess boundary (max recoverable per velocity level):")
    for i, vel in enumerate(vel_offsets):
        succ_row = succ_grid[i, :]
        if succ_row.any():
            max_pos = pos_offsets[np.where(succ_row)[0][-1]]
            print(f"  vel={vel:4.2f} m/s -> success up to pos={max_pos:6.1f} m")
        else:
            print(f"  vel={vel:4.2f} m/s -> no success at any position")

    # For each position column, find the max velocity that still succeeds
    print("\nCritical velocity (max recoverable per position offset):")
    for j, pos in enumerate(pos_offsets):
        succ_col = succ_grid[:, j]
        if succ_col.any():
            max_vel = vel_offsets[np.where(succ_col)[0][-1]]
            print(f"  pos={pos:6.1f} m -> success up to vel={max_vel:4.2f} m/s")


def plot_envelope(pos_offsets, vel_offsets, prop_grid, succ_grid):
    fig, axs = plt.subplots(1, 2, figsize=(7,3), tight_layout=True)

    im = axs[0].pcolormesh(pos_offsets, vel_offsets, prop_grid,
                           shading="auto", cmap="rainbow")
    
    # Overlay 5000 propellant limit contour line
    CS = axs[0].contour(pos_offsets, vel_offsets, prop_grid, 
                        levels=[PROPELLANT_BUDGET], colors='k', linewidths=1.5)
    axs[0].clabel(CS, inline=True, fmt={PROPELLANT_BUDGET: "5000 limit"}, fontsize=8)

    fig.colorbar(im, ax=axs[0], label="propellant proxy")
    axs[0].set_xlabel("initial position offset [m]")
    axs[0].set_ylabel("initial velocity offset [m/s]")

    binary_cmap = ListedColormap(["#e74c3c", "#2ecc71"])
    axs[1].pcolormesh(pos_offsets, vel_offsets, succ_grid,
                      shading="auto", cmap=binary_cmap, vmin=0, vmax=1)
    axs[1].set_xlabel("initial position offset [m]")
    axs[1].set_ylabel("initial velocity offset [m/s]")
    axs[1].set_title("Docking success envelope", fontsize=10)
    axs[1].set_box_aspect(1.0)
    axs[1].text(75, 2, "SUCCESS", color="k", ha="center", va="center", fontsize=8)
    axs[1].text(75, 4, "FAILED", color="k",ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/envelope_results_2.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()