"""LQR tuning: sweep the control weight R and quantify the tradeoff
between convergence speed and propellant use"""

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
from scripts.run_nominal import build_params


def run_one(params, R_scale, dt=1.0, t_end=3000.0):
    """Run a single docking sim (basically a nominal run) with translation R scaled by R_scale.
    Returns (settling_time, propellant_proxy, max_saturation_fraction)."""

    # same settings as run_nominal.py, except R_trans is scaled by R_scale
    Q_trans = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    R_trans = np.diag([1.0, 1.0, 1.0]) * R_scale
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.diag([1.0, 1.0, 1.0]) * 100.0

    x0 = np.zeros(13)
    x0[0:3] = [50.0, -100.0, 20.0]
    x0[6:10] = [0.0872, 0.0, 0.0, 0.9962]
    dock_position = np.array([0.0, -5.0, 0.0])

    plant = Plant(params, x0)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(0.1, 0.01, np.deg2rad(0.5), np.deg2rad(0.05), seed=42)
    nav = PassthroughNav()
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)
    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators,
                    scheduler)

    log = sim.run(t_end=t_end)

    pos_err = np.linalg.norm(log["x_truth"][:, 0:3] - dock_position, axis=1)

    # Settling time: last time position error exceeds 1 m threshold
    threshold = 1.0
    above = np.where(pos_err > threshold)[0]
    settling_time = log["t"][above[-1]] if len(above) else 0.0

    # Propellant proxy: integral of |thrust| over time
    propellant = np.sum(np.abs(log["u_applied"][:, 0:3])) * dt

    # Saturation fraction (translation channels)
    sat_frac = log["saturated"][:, 0:3].mean()

    return settling_time, propellant, sat_frac


def main():
    params = build_params()
    R_scales = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]  # arbitrary range

    results = []
    for R in R_scales:
        st, prop, sat = run_one(params, R)
        results.append((R, st, prop, sat))
        print(f"R={R:6.1f} | settling={st:7.1f}s | "
              f"propellant={prop:9.1f} | saturation={sat*100:5.1f}%")

    results = np.array(results)

    # Tradeoff curve: settling time vs propellant
    plt.figure(figsize=(3,3))
    plt.plot(results[:, 2], results[:, 1], "o-")
    for R, st, prop, _ in results:
        plt.annotate(f"R={R:.0f}", (prop, st),
                     textcoords="offset points", xytext=(0,0))
    plt.xlabel("propellant proxy (∫|thrust| dt)")
    plt.ylabel("settling time [s]")
    plt.title("LQR tradeoff: aggressive (low R) \n↔ economical (high R)")
    plt.savefig("results/lqr_tuning.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
