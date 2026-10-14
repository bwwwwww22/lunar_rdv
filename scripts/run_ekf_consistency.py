"""Run EKF-based sim while logging NEES & NIS and plot the
 statistics against chi^2 confidence bounds.

Here we expect 12 for NEES (whether the actual state error
is consistent with the filter's claimed covariance P) and NIS 
(whether the innovation is consistent with S), expected to be 3 per sensor
(star tracker, lidar); the gyro is a predict input, so it has no innovation.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.ekf import MEKF
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.environment.actuators import Actuators
from src.analysis.consistency import nees, nis, chi2_bounds, compute_error_state
from scripts.run_nominal import build_params
from src.utils.quaternions import quat_multiply, quat_normalize
from src.analysis.monte_carlo import (default_sensor_config, default_R,
        make_sensor_suite, default_ekf_config)


def run_with_consistency_logging(t_end=3000.0, dt=1.0, seed=42, q_trans_scale=1e-3,
                                 matched_noise=False):
    """Manually drive the loop (bypassing SimRunner) to get access to the
    EKF's innovation y and covariance S at each step for NIS."""
    params = build_params()

    Q_trans = np.diag([1.0, 1.0, 1.0,
                       0.1, 0.1, 0.1])
    R_trans = np.eye(3) * 10.0
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.eye(3) * 100.0

    # truth sensor noise and the matching EKF R/P0/Q (all from one config)
    cfg = default_sensor_config()
    R_ekf = default_R(cfg)
    P0, Q_ekf = default_ekf_config(cfg, q_trans_scale=q_trans_scale)

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = make_sensor_suite(dt, seed, cfg)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)

    x0_est = x0.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] = 0.0   # filter state 10:13 holds the gyro-bias estimate b_hat, not w
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(dq, x0_est[6:10]))

    if matched_noise:
        # Validation: give the truth exactly the process noise the filter assumes
        # (diagonal Q_d = Q*dt on [r, v]). If the covariance bookkeeping is right,
        # every NEES block should now land near 3.
        plant = Plant(params, x0, seed=seed + 7,
                      process_noise_std=np.sqrt(np.diag(Q_ekf)[0:6] * dt))
    ekf = MEKF(params, x0_est, P0, Q_ekf, R_ekf)

    n_steps = int(t_end / dt)
    t_hist = np.zeros(n_steps + 1)
    nees_hist = np.zeros(n_steps + 1)
    nis_star = np.full(n_steps + 1, np.nan)    # NaN on steps where that sensor had no solution
    nis_lidar = np.full(n_steps + 1, np.nan)
    blocks = {"pos": slice(0, 3), "vel": slice(3, 6), "att": slice(6, 9), "bias": slice(9, 12)}
    nees_blk = {name: np.zeros(n_steps + 1) for name in blocks}   # marginal NEES, 3 dof each
    truth_hist = np.zeros((n_steps + 1, 13))
    est_hist = np.zeros((n_steps + 1, 13))

    truth_hist[0] = plant.get_state()
    est_hist[0] = ekf.get_estimate()

    u_applied = np.zeros(6)
    for k in range(n_steps):
        t = k * dt
        # Sensor first (the gyro reading drives predict), then Pred + Update,
        # same order as EKFNav.estimate
        z = sensor.measure(plant.get_state())
        if k == 0:      # x0_est is already the t_0 estimate; same rule as EKFNav
            ekf.w_m = np.asarray(z["gyro"], dtype=float)
        else:
            ekf.predict(u_applied, z["gyro"], dt)
        results = ekf.update(z)        # [(name, y, S), ...] for the sensors fused this step

        # Log NEES (12 dof; needs the FILTER state [r,v,q,b_hat] and the true gyro bias)
        nees_hist[k + 1] = nees(plant.get_state(), sensor.gyro.bias,
                                ekf.get_filter_state(), ekf.get_covariance())
        # Log NEES per block, using the marginal covariance P[sl, sl] (3 dof each)
        dx = compute_error_state(plant.get_state(), sensor.gyro.bias, ekf.get_filter_state())
        P = ekf.get_covariance()
        for name, sl in blocks.items():
            nees_blk[name][k + 1] = dx[sl] @ np.linalg.solve(P[sl, sl], dx[sl])
        # Log NIS per sensor (3 dof each)
        for name, y, S in results:
            (nis_star if name == "star" else nis_lidar)[k + 1] = nis(y, S)

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
    return (t_hist[1:], nees_hist[1:], nis_star[1:], nis_lidar[1:],
            {name: h[1:] for name, h in nees_blk.items()})


def plot_consistency(t, nees_h, nis_star, nis_lidar, nees_blk=None,
                     fname="results/ekf_consistency.png", n_runs=1):
    """NEES (12 dof) and per-sensor NIS (3 dof each) with chi^2 bounds.
    n_runs > 1: inputs are averages over n_runs seeds, so the bounds are the tighter
    ones for a mean of n_runs chi^2 samples."""
    panels = [("NEES", nees_h, 12, "State-error consistency"),
              ("NIS star tracker", nis_star, 3, "Innovation consistency: star tracker"),
              ("NIS lidar", nis_lidar, 3, "Innovation consistency: lidar")]

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True, tight_layout=True)
    ss = slice(int(0.6 * len(t)), len(t))   # steady state = last 40% of run
    print("\n=== Consistency summary (steady state) ===")

    for ax, (name, h, dof, title) in zip(axs, panels):
        lo, hi = chi2_bounds(dof, N_runs=n_runs, alpha=0.05)
        ax.plot(t, h, lw=0.5, alpha=0.7, label=name)
        ax.axhline(dof, color="k", ls=":", label=f"expected = {dof}")
        ax.axhline(lo, color="tab:orange", ls="--", label="95% bounds")
        ax.axhline(hi, color="tab:orange", ls="--")
        ax.set_ylabel(name); ax.set_title(title); ax.legend()

        h_ss = h[ss][~np.isnan(h[ss])]      # drop steps where the sensor had no solution
        frac = np.mean((h_ss >= lo) & (h_ss <= hi))
        print(f"{name:17s} mean: {h_ss.mean():6.2f} (expected {dof}), "
              f"{frac:.2%} inside 95% bounds")

    if nees_blk is not None:
        lo, hi = chi2_bounds(3, N_runs=n_runs, alpha=0.05)
        print("--- NEES per block (3 dof each, expected 3) ---")
        for name, h in nees_blk.items():
            h_ss = h[ss]
            print(f"NEES {name:5s}        mean: {h_ss.mean():6.2f}, "
                  f"{np.mean((h_ss >= lo) & (h_ss <= hi)):.2%} inside 95% bounds")

    axs[-1].set_xlabel("time [s]")
    plt.savefig(fname, dpi=300)


def main(matched_noise=False):
    t, nees_h, nis_star, nis_lidar, nees_blk = run_with_consistency_logging(
        matched_noise=matched_noise)
    fname = ("results/ekf_consistency_matched.png" if matched_noise
             else "results/ekf_consistency.png")
    print("MATCHED process noise in truth" if matched_noise else "Nominal truth (no process noise)")
    plot_consistency(t, nees_h, nis_star, nis_lidar, nees_blk, fname=fname)

def main_multi(n_seeds=30, matched_noise=False):
    """Average NEES/NIS over many seeds at each timestep, so slow states (bias) aren't
    judged from a single run. Each seed gets its own true bias and noise draws."""
    runs = [run_with_consistency_logging(seed=42 + i, matched_noise=matched_noise)
            for i in range(n_seeds)]
    t = runs[0][0]
    nees_h = np.mean([r[1] for r in runs], axis=0)
    nis_star = np.nanmean([r[2] for r in runs], axis=0)
    nis_lidar = np.nanmean([r[3] for r in runs], axis=0)
    nees_blk = {name: np.mean([r[4][name] for r in runs], axis=0) for name in runs[0][4]}
    fname = "results/ekf_consistency_multi" + ("_matched" if matched_noise else "") + ".png"
    print(f"Averaged over {n_seeds} seeds" + (", MATCHED process noise" if matched_noise else ""))
    plot_consistency(t, nees_h, nis_star, nis_lidar, nees_blk, fname=fname, n_runs=n_seeds)


if __name__ == "__main__":
    import sys
    matched = "--matched" in sys.argv
    if "--multi" in sys.argv:
        main_multi(matched_noise=matched)
    else:
        main(matched_noise=matched)