"""Run EKF-based sim while logging NEES & NIS and plot the
 statistics against chi^2 confidence bounds.

Here we expect 12 for both NEES (whether the actual state error
is consistent with the filter's claimed covariance P) and NIS 
(whether the innovation is consistent with S).
"""

import numpy as np
import matplotlib.pyplot as plt
from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor
from src.fsw.ekf import MEKF
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.environment.actuators import Actuators
from src.analysis.consistency import nees, nis, chi2_bounds
from scripts.run_nominal import build_params
from src.utils.quaternions import quat_multiply, quat_normalize


def run_with_consistency_logging(t_end=3000.0, dt=1.0, seed=42, q_scale=0.01):
    """Manually drive the loop (bypassing SimRunner) to get access to the
    EKF's innovation y and covariance S at each step for NIS."""
    params = build_params()

    Q_trans = np.diag([1.0, 1.0, 1.0,
                       0.1, 0.1, 0.1])
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.eye(3) * 100.0
    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)
    P0 = np.diag([
            1.0, 1.0, 1.0, 
            0.1, 0.1, 0.1, 
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2,
            np.deg2rad(0.1)**2, np.deg2rad(0.1)**2, np.deg2rad(0.1)**2,
    ])
    Q_base = np.diag([
        1e-4, 1e-4, 1e-4,
        1e-3, 1e-3, 1e-3,
        1e-6, 1e-6, 1e-6,
        1e-6, 1e-6, 1e-6,
    ])
    R_ekf = np.diag(
        [pos_std**2]*3 + [vel_std**2]*3 +
        [att_std**2]*3 + [rate_std**2]*3
    )

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.9962, 0.0872, 0.0, 0.0]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_std, vel_std, att_std, rate_std, seed=seed)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)

    x0_est = x0.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] += np.deg2rad(np.array([0.1, -0.1, 0.1]))
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(x0_est[6:10], dq))

    ekf = MEKF(params, x0_est, P0, Q_base * q_scale, R_ekf)

    n_steps = int(t_end / dt)
    t_hist = np.zeros(n_steps + 1)
    nees_hist = np.zeros(n_steps + 1)
    nis_hist = np.zeros(n_steps + 1)
    truth_hist = np.zeros((n_steps + 1, 13))
    est_hist = np.zeros((n_steps + 1, 13))

    truth_hist[0] = plant.get_state()
    est_hist[0] = ekf.get_estimate()

    u_applied = np.zeros(6)
    for k in range(n_steps):
        t = k * dt
        # Pred
        ekf.predict(u_applied, dt)
        # Sensor + Update
        z = sensor.measure(plant.get_state())
        y, S = ekf.update(z)

        # Log NEES/NIS
        nees_hist[k + 1] = nees(plant.get_state(), ekf.get_estimate(),ekf.get_covariance())
        nis_hist[k + 1] = nis(y, S)

        # Ctrl
        x_ref = guidance.reference(t, ekf.get_estimate())
        u_cmd = controller.command(ekf.get_estimate(), x_ref)
        u_applied = actuators.apply(u_cmd)

        # Advance the plant
        plant.step(u_applied, dt)

        t_hist[k + 1] = t + dt
        truth_hist[k + 1] = plant.get_state()
        est_hist[k + 1] = ekf.get_estimate()

    # drop NEES/NIS at k=0 (meaningless bc no update yet)
    return t_hist[1:], nees_hist[1:], nis_hist[1:]


def plot_consistency(t, nees_h, nis_h, dof=12):
    """Plot NEES and NIS with single-run chi^2 bounds."""
    lo, hi = chi2_bounds(dof, N_runs=1, alpha=0.05)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True, tight_layout=True)

    axs[0].plot(t, nees_h, lw=0.5, alpha=0.7, label="NEES")
    axs[0].axhline(dof, color="k", ls=":", label=f"expected = {dof}")
    axs[0].axhline(lo, color="tab:orange", ls="--", label="95% bounds")
    axs[0].axhline(hi, color="tab:orange", ls="--")
    axs[0].set_ylabel("NEES"); axs[0].set_title("State-error consistency")
    axs[0].legend(); 

    axs[1].plot(t, nis_h, lw=0.5, alpha=0.7, label="NIS")
    axs[1].axhline(dof, color="k", ls=":")
    axs[1].axhline(lo, color="tab:orange", ls="--")
    axs[1].axhline(hi, color="tab:orange", ls="--")
    axs[1].set_ylabel("NIS"); axs[1].set_xlabel("time [s]")
    axs[1].set_title("Innovation consistency"); 

    # print summary stats
    ss = slice(int(0.6 * len(t)), len(t))
    print("\n=== Consistency summary (steady state) ===")
    print(f"NEES mean: {nees_h[ss].mean():.2f}  (expected {dof})")
    print(f"NIS  mean: {nis_h[ss].mean():.2f}  (expected {dof})")
    frac_in_nees = np.mean((nees_h[ss] >= lo) & (nees_h[ss] <= hi))
    frac_in_nis  = np.mean((nis_h[ss] >= lo)  & (nis_h[ss] <= hi))
    print(f"NEES fraction inside 95% bounds: {frac_in_nees:.2%}")
    print(f"NIS  fraction inside 95% bounds: {frac_in_nis:.2%}")

    plt.savefig("results/ekf_consistency.png", dpi=300)


def main():
    t, nees_h, nis_h = run_with_consistency_logging()
    plot_consistency(t, nees_h, nis_h)

if __name__ == "__main__":
    main()