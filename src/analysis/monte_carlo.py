"""Monte Carlo harness: run N docking sims with dispersed initial conditions
and independent sensor noise seeds. Aggregates docking success and
performance statistics for robustness analysis."""

import numpy as np
from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import PoseSensor
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error, quat_multiply, quat_normalize


def default_ekf_config():
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
    Q = np.diag([
        1e-4, 1e-4, 1e-4,            # pos process noise
        1e-3, 1e-3, 1e-3,            # velocity
        1e-6, 1e-6, 1e-6,            # attitude (small)
        1e-6, 1e-6, 1e-6,            # angular rate
    ])
    return P0, Q


def sample_dispersed_x0(rng, nominal_x0,
                        pos_std=50.0, vel_std=2.0, att_std_deg=30.0, rate_std_dps=1.0):
    """dispersed initial condition around a nominal x0 (drawn from Gaussian)"""
    x0 = nominal_x0.copy()
    x0[0:3]   += rng.normal(0, pos_std, 3)
    x0[3:6]   += rng.normal(0, vel_std, 3)

    # Small-angle attitude dispersion applied multiplicatively
    # cannot simply add Gaussian noise to a 4D quaternion vector without breaking its normalization
    # a random 3D small-angle rotation (δθ) is converted into a delta-quaternion (δq) 
    # and multiplied onto the nominal orientation
    dtheta = rng.normal(0, np.deg2rad(att_std_deg), 3)
    dq = np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]])
    dq = quat_normalize(dq)
    x0[6:10] = quat_normalize(quat_multiply(x0[6:10], dq))

    x0[10:13] += rng.normal(0, np.deg2rad(rate_std_dps), 3)
    return x0


def run_single(seed, nominal_x0, params, dock_position, t_end=3000.0, dt=1.0,
               disperse_ic=True):
    """Run one closed-loop docking with given seed. Returns metrics dict."""
    rng = np.random.default_rng(seed)

    # LQR (fixed)
    Q_trans = np.diag([1.0, 1.0, 1.0, 
                        0.1, 0.1, 0.1])
    R_trans = np.diag([1.0, 1.0, 1.0]) * 10.0 
    Q_att   = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_att   = np.diag([1.0, 1.0, 1.0]) * 100.0

    pos_std, vel_std = 0.1, 0.01
    att_std, rate_std = np.deg2rad(0.5), np.deg2rad(0.05)
    R_ekf = np.diag(
        [pos_std**2]*3 + [vel_std**2]*3 +
        [att_std**2]*3 + [rate_std**2]*3
    )

    P0, Q_ekf = default_ekf_config()

    # Dispersed truth IC
    x0_truth = (sample_dispersed_x0(rng, nominal_x0) if disperse_ic
                else nominal_x0.copy())

    # --- EKF initial estimate: a further small offset from the dispersed truth
    # simulates real-life uncertainty at initialization: the EKF does not 
    # know the exact true position of the spacecraft right at t=0; 
    # it starts with a small initial estimate error (up to 0.5 m) and relies on
    # its measurement updates to converge
    x0_est = x0_truth.copy()
    x0_est[0:3] += rng.normal(0, 0.5, 3)

    plant = Plant(params, x0_truth)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = PoseSensor(pos_std, vel_std, att_std, rate_std, seed=seed + 1)
    nav = EKFNav(params, x0_est, P0, Q_ekf, R_ekf)
    guidance = Guidance(dock_position=dock_position)
    controller = LQRController(params, Q_trans, R_trans, Q_att, R_att)
    scheduler = Scheduler(dt=dt)
    sim = SimRunner(plant, sensor, nav, guidance, controller, actuators,
                    scheduler)
    log = sim.run(t_end=t_end)

    # Metrics
    truth = log["x_truth"]
    final_pos_err = np.linalg.norm(truth[-1, 0:3] - dock_position)
    final_att_err = np.linalg.norm(quat_error(truth[-1, 6:10],
                                              np.array([1, 0, 0, 0])))
    max_sat = log["saturated"].mean(axis=0).max()
    propellant = np.sum(np.abs(log["u_applied"][:, 0:3])) * dt

    # Docking success criterion
    #success = (final_pos_err < 1.0) and (np.rad2deg(final_att_err) < 5.0)
    # Add a propellant budget as a success criterion bc I was getting 100% success
    PROPELLANT_BUDGET = 5000 # between nominal ~2700 and stressed ~6700
    success = (final_pos_err < 1.0
           and np.rad2deg(final_att_err) < 5.0
           and propellant < PROPELLANT_BUDGET)

    return {
        "seed": seed,
        "final_pos_err_m": final_pos_err,
        "final_att_err_deg": np.rad2deg(final_att_err),
        "max_sat_fraction": max_sat,
        "propellant": propellant,
        "success": success,
    }


def run_monte_carlo(n_runs, nominal_x0, params, dock_position, **kwargs):
    """Run N sims with different seeds. Returns list of metric dicts."""
    results = []
    for i in range(n_runs):
        r = run_single(seed=1000 + i, nominal_x0=nominal_x0, params=params,
                       dock_position=dock_position, **kwargs)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  completed {i+1}/{n_runs}")
    return results


def summarize(results):
    n = len(results)
    pos_errs = np.array([r["final_pos_err_m"] for r in results])
    att_errs = np.array([r["final_att_err_deg"] for r in results])
    props    = np.array([r["propellant"] for r in results])
    sats     = np.array([r["max_sat_fraction"] for r in results])
    succ     = np.array([r["success"] for r in results])

    print(f"Success rate: {succ.mean()*100:.1f}%")
    print(f"Final position error [m]:  mean={pos_errs.mean():.3f}, "
          f"95th={np.percentile(pos_errs, 95):.3f}, max={pos_errs.max():.3f}")
    print(f"Final attitude error [deg]: mean={att_errs.mean():.3f}, "
          f"95th={np.percentile(att_errs, 95):.3f}, max={att_errs.max():.3f}")
    print(f"Propellant proxy: mean={props.mean():.1f}, max={props.max():.1f}")
    print(f"Max saturation fraction:  mean={sats.mean()*100:.1f}%, "
          f"max={sats.max()*100:.1f}%")