"""Propagation w/ zero control (just dynamics). For verification"""

import numpy as np
import matplotlib.pyplot as plt

from src.environment.plant import Plant

def build_params():
    # Lunar orbit rate: n = sqrt(mu / a^3)
    mu_moon = 4.9028e12          # m^3/s^2
    altitude = 100e3             # 100 km
    r_moon = 1.7374e6            # m
    a = r_moon + altitude
    n = np.sqrt(mu_moon / a**3)  # rad/s

    inertia = np.diag([100.0, 120.0, 80.0])  # kg·m^2, arbitrary values for chaser
    return {
        "mass": 500.0,                       # kg, arbitrary chaser mass
        "inertia": inertia,
        "inertia_inv": np.linalg.inv(inertia),
        "orbit_rate": n,
    }


def run_openloop(params, x0, dt, t_end):
    plant = Plant(params, x0)
    n_steps = int(t_end / dt)
    t_hist = np.zeros(n_steps + 1)
    x_hist = np.zeros((n_steps + 1, 13))
    x_hist[0] = plant.get_state() # init

    u = np.zeros(6)  # zero control
    for k in range(n_steps):
        x_hist[k + 1] = plant.step(u, dt) # step the plant fwd
        t_hist[k + 1] = (k + 1) * dt
    return t_hist, x_hist


def main():
    params = build_params()
    n = params["orbit_rate"]

    # Initial condition chosen to give a bounded/closed CW relative orbit,
    # which requires vy0 = -2*n*x0 to prevent secular drift in y
    x0_pos = np.array([1000.0, 0.0, 500.0])           # r [m]
    x0_vel = np.array([0.0, -2 * n * x0_pos[0], 0.0]) # v [m/s]
    q0 = np.array([0.0, 0.0, 0.0, 1.0])               # identity
    w0 = np.array([0.0, 0.01, 0.0])                   # small spin [rad/s]
    x0 = np.concatenate([x0_pos, x0_vel, q0, w0])

    period = 2 * np.pi / n
    t_hist, x_hist = run_openloop(params, x0, dt=1.0, t_end=2 * period)

    # Plot in-plane relative traj
    plt.figure()
    plt.plot(x_hist[:, 1], x_hist[:, 0])  # y (along-track) vs x (radial)
    plt.xlabel("along-track y [m]"); plt.ylabel("radial x [m]")
    plt.title("CW relative motion (should be a closed ellipse)")
    plt.axis("equal");

    # Quaternion norm over time
    qnorm = np.linalg.norm(x_hist[:, 6:10], axis=1)
    plt.figure()
    plt.plot(t_hist, qnorm - 1.0)
    plt.xlabel("time [s]"); plt.ylabel("‖q‖ − 1")
    plt.title("Quaternion norm error (should stay ~0)")

    plt.show()


if __name__ == "__main__":
    main()
