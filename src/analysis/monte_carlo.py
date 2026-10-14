"""Monte Carlo harness: run N docking sims with dispersed initial conditions
and independent sensor noise seeds. Aggregates docking success and
performance statistics for robustness analysis."""

import numpy as np
from src.environment.plant import Plant
from src.environment.actuators import Actuators
from src.fsw.sensors import LidarSensor, GyroSensor, StarTracker, SensorSuite
from src.fsw.navigation import EKFNav
from src.fsw.guidance import Guidance
from src.fsw.control import LQRController
from src.sim.scheduler import Scheduler
from src.sim.runner import SimRunner
from src.utils.quaternions import quat_error, quat_multiply, quat_normalize

# TODO: search for real numbers
ARCSEC = np.deg2rad(1.0 / 3600.0)           # arcsec -> rad
ARW_UNIT = np.deg2rad(1.0) / 60.0           # deg/sqrt(hr) -> rad/sqrt(s)
RRW_UNIT = np.deg2rad(1.0) / 3600.0 / 60.0  # deg/hr/sqrt(hr) -> rad/s^1.5
BIAS_UNIT = np.deg2rad(1.0) / 3600.0        # deg/hr -> rad/s

def default_sensor_config():
    """Truth sensor noise (1-sigma). Currently use PLACEHOLDER values.
    Single source of truth: builds the sensors AND sets the EKF's R, P0 and Q """
    return {
        # gyro
        "arw": 0.5 * ARW_UNIT,            # rate noise sigma_rc [rad/sqrt(s)]
        "rrw": 0.5 * RRW_UNIT,            # bias random walk sigma_wc [rad/s^1.5]
        "bias0_std": 10.0 * BIAS_UNIT,    # turn-on bias [rad/s]
        # star tracker
        "star_cross": 2.0 * ARCSEC,       # [rad]
        "star_bore": 10.0 * ARCSEC,       # [rad]
        "star_hz": 1.0,
        # lidar
        "lidar_range": 0.1,               # [m]
        "lidar_angle": np.deg2rad(0.1),   # azimuth & elevation [rad]
        "lidar_hz": 1.0,
    }

def default_R(cfg=None):
    """Per-sensor measurement covariances for the EKF
    p.s. same sigmas that the sensors use """

    cfg = cfg or default_sensor_config()
    return {
        "star": np.diag([cfg["star_cross"]**2, cfg["star_cross"]**2, 
                         cfg["star_bore"]**2]),
        "lidar": np.diag([cfg["lidar_range"]**2, cfg["lidar_angle"]**2, 
                          cfg["lidar_angle"]**2]),
    }

def make_sensor_suite(dt, seed, cfg=None):
    """Gyro + star tracker + lidar"""
    cfg = cfg or default_sensor_config()
    s_gyro, s_star, s_lidar = np.random.SeedSequence(seed).spawn(3)
    return SensorSuite(
        GyroSensor(cfg["arw"], cfg["rrw"], cfg["bias0_std"], 
                   dt, seed=s_gyro),
        StarTracker(cfg["star_cross"], cfg["star_bore"], cfg["star_hz"], 
                    dt, seed=s_star),
        LidarSensor(cfg["lidar_range"], cfg["lidar_angle"], cfg["lidar_hz"], 
                    dt, seed=s_lidar),
    )


def default_ekf_config(cfg=None, q_trans_scale=1e-3):
    """Error state [δr, δv, δθ, Δb]; Q is continuous (predict multiplies by dt).
    q_trans_scale scales only the position/velocity blocks of Q: the deliberate margin
    for unmodeled translational effects (chosen from tune_ekf.py). The gyro blocks come
    from the sensor config and are never scaled. Pass 1.0 to get the unscaled baseline."""
    cfg = cfg or default_sensor_config()
    P0 = np.diag([
        1.0, 1.0, 1.0,               # initial pos uncertainty (m^2)
        0.1, 0.1, 0.1,               # velocity  (m/s)^2
        np.deg2rad(5)**2,            # attitude (rad^2), ~5 deg 1-sigma
        np.deg2rad(5)**2,
        np.deg2rad(5)**2,
        cfg["bias0_std"]**2,         # init bias uncertainty = turn-on bias var. since b_hat0 = 0
        cfg["bias0_std"]**2,
        cfg["bias0_std"]**2,
    ])
    Q = np.diag([
        1e-4, 1e-4, 1e-4,            # pos process noise
        1e-3, 1e-3, 1e-3,            # velocity
        cfg["arw"]**2, cfg["arw"]**2, cfg["arw"]**2,   # gyro noise sigma_r^2
        cfg["rrw"]**2, cfg["rrw"]**2, cfg["rrw"]**2,   # bias rand walk sigma_w^2
    ])
    Q[0:6, 0:6] *= q_trans_scale
    return P0, Q


# these stds disperse the TRUE initial state (how far off the chaser starts), which are
# INDEPENDENT of sensor noise, which comes from default_sensor_config()
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
    dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
    dq = quat_normalize(dq)
    x0[6:10] = quat_normalize(quat_multiply(dq, x0[6:10]))

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

    # truth sensor noise and the matching EKF R/P0/Q (all from one config)
    cfg = default_sensor_config()
    R_ekf = default_R(cfg)
    P0, Q_ekf = default_ekf_config(cfg)

    # Dispersed truth IC
    x0_truth = (sample_dispersed_x0(rng, nominal_x0) if disperse_ic
                else nominal_x0.copy())

    # EKF initial estimate
    x0_est = x0_truth.copy()
    x0_est[0:3]   += np.array([1.0, -1.0, 1.0])
    x0_est[3:6]   += np.array([0.316, -0.316, 0.316])
    x0_est[10:13] = 0.0   # filter state 10:13 hold the b_hat (gyro bias est.), not w
    dtheta = np.deg2rad(np.array([5.0, -5.0, 5.0]))
    dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
    dq = quat_normalize(dq)
    x0_est[6:10] = quat_normalize(quat_multiply(dq, x0_est[6:10]))

    plant = Plant(params, x0_truth)
    actuators = Actuators(max_force=10.0, max_torque=5.0)
    sensor = make_sensor_suite(dt, seed + 1, cfg)
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
    final_att_err = np.linalg.norm(quat_error(truth[-1, 6:10], np.array([0, 0, 0, 1])))
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

    # convergence-only success (position < 1 m and attitude < 5 deg), ignoring the propellant budget
    conv = (pos_errs < 1.0) & (att_errs < 5.0)
    print(f"Success rate, no propellant constraint: {conv.mean()*100:.1f}%")
    print(f"Success rate, within propellant budget: {succ.mean()*100:.1f}%")
    print(f"Final position error [m]:  mean={pos_errs.mean():.3f}, "
          f"95th={np.percentile(pos_errs, 95):.3f}, max={pos_errs.max():.3f}")
    print(f"Final attitude error [deg]: mean={att_errs.mean():.3f}, "
          f"95th={np.percentile(att_errs, 95):.3f}, max={att_errs.max():.3f}")
    print(f"Propellant proxy: mean={props.mean():.1f}, max={props.max():.1f}")
    print(f"Max saturation fraction:  mean={sats.mean()*100:.1f}%, "
          f"max={sats.max()*100:.1f}%")