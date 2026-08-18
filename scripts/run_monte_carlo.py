"""N Monte Carlo docking sims with dispersed initial conditions and 
independent noise seeds; reports success rate, error percentiles, 
propellant usage, and saturation stats."""

import numpy as np
import matplotlib.pyplot as plt

from src.analysis.monte_carlo import run_monte_carlo, summarize
from scripts.run_nominal import build_params


def main():
    params = build_params()

    nominal_x0 = np.zeros(13)
    nominal_x0[0:3] = [50.0, -100.0, 20.0]
    nominal_x0[6:10] = [0.9962, 0.0872, 0.0, 0.0]

    dock_position = np.array([0.0, -5.0, 0.0])

    N = 50   # maybe bump to 200-500 later
    print(f"Running {N} Monte Carlo docking sims...")
    results = run_monte_carlo(N, nominal_x0, params, dock_position)
    summarize(results)

    pos_errs = np.array([r["final_pos_err_m"] for r in results])
    att_errs = np.array([r["final_att_err_deg"] for r in results])
    props    = np.array([r["propellant"] for r in results])
    succ     = np.array([r["success"] for r in results])

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
    axs[0].hist(pos_errs, bins=20, edgecolor="k", linewidth=0.25)
    #axs[0].axvline(1.0, color="r", ls="--", label="success threshold (1 m)")
    axs[0].set_xlabel("final position error [m]")
    axs[0].set_ylabel("count")
    axs[0].set_title("Final position error")

    axs[1].hist(att_errs, bins=20, edgecolor="k", linewidth=0.25)
    #axs[1].axvline(5.0, color="r", ls="--", label="success threshold (5°)")
    axs[1].set_xlabel("final attitude error [deg]")
    axs[1].set_title("Final attitude error")

    axs[2].hist(props, bins=20, edgecolor="k")
    axs[2].set_xlabel("propellant proxy (∫|thrust| dt)")
    axs[2].set_title("Propellant usage")
    for ax in axs: ax.set_box_aspect(1.0)

    fig.suptitle(f"Monte Carlo results — {len(results)} runs, "
                 f"{succ.mean()*100:.0f}% success rate")
    plt.tight_layout()
    plt.savefig("results/monte_carlo_results.png", dpi=300)


if __name__ == "__main__":
    main()