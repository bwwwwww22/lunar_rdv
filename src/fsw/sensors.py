"""Synthetic sensor model (13-state)"""

import numpy as np

from src.utils.quaternions import quat_multiply, quat_normalize, quat_to_rotmat

class PoseSensor:
    def __init__(self, pos_noise_std, vel_noise_std, att_noise_std,
                 rate_noise_std, seed=None):
        """Noise std devs:
        pos [m], vel [m/s], att [rad] (small-angle), rate [rad/s]."""
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.att_noise_std = att_noise_std
        self.rate_noise_std = rate_noise_std
        self.rng = np.random.default_rng(seed) # initialize generator

    def measure(self, x_truth):
        """Return a noisy (RNG) 13-state measurement of the truth state."""
        z = x_truth.copy() # COPY!

        # Position, velocity & angular rate: additive Gaussian noise
        z[0:3] += self.rng.normal(0, self.pos_noise_std, 3)
        z[3:6] += self.rng.normal(0, self.vel_noise_std, 3)
        z[10:13] += self.rng.normal(0, self.rate_noise_std, 3)

        # Attitude: perturb by a small-angle error quaternion
        # because adding noise to quaternion components would break the norm
        # dθ = [dθx, dθy, dθz] small-angle error
        dtheta = self.rng.normal(0, self.att_noise_std, 3)
        # dq = [sin(|dθ|/2) * (dθ/|dθ|), cos(|dθ|/2)] ≈ [0.5*dθ, 1] for small |dθ| (scalar-last)
        dq = np.array([0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2], 1.0])
        dq = quat_normalize(dq) # don't forget to normalize again
        z[6:10] = quat_normalize(quat_multiply(dq, x_truth[6:10]))

        return z

# To support testing of EKF robustness to sensor dropout, i wrapped our sensor class
# in a DropoutSensor that returns None during a specified time window
class DropoutSensor:
    """Wraps a sensor to simulate measurement dropout over time
    window(s). Returns None during dropout."""

    def __init__(self, inner_sensor, dropout_windows, dt):
        """
        inner_sensor    : the underlying sensor (e.g., PoseSensor)
        dropout_windows : list of (t_start, t_end) tuples [s]
        dt              : timestep
        """
        self.inner = inner_sensor
        self.dropout_windows = dropout_windows
        self.dt = dt # to track time
        self.t = 0.0

    def measure(self, x_truth):
        self.t += self.dt
        for (t0, t1) in self.dropout_windows:
            if t0 <= self.t <= t1:
                return None          # no measurement during dropout
        return self.inner.measure(x_truth)

# 10/02
def lidar_model(r_hill, q):
    """ rho, azimuth & elevation of the tgt, seen from the chaser (sensor = body frame)
    hill is chaser-relative-to-target, so the line of sight is -negative of that
    Also returns the body-frame line of sight p, which the EKF needs for Jacobian."""
    p = quat_to_rotmat(q) @ (-np.asarray(r_hill, dtype=float))
    rho = np.linalg.norm(p)      # rho = ||p||
    az = np.arctan2(p[1], p[0])  # az = atan2(y, x)
    el = np.arcsin(p[2] / rho)   # el = asin(z/ρ), measured up from the xy plane
    return np.array([rho, az, el]), p

class LidarSensor:
    def __init__(self, range_std, angle_std, rate_hz, dt, seed=42):
        self.stds = np.array([range_std, angle_std, angle_std])
        self.period = max(1, round(1.0 / (rate_hz * dt)))
        self.rng = np.random.default_rng(seed)
        self.k = 0

    def measure(self, x_truth):
        """Noisy [rho, az, el] in the sensor frame, or none if blinded"""
        self.k += 1
        if self.k % self.period:
            return None
        z, _ = lidar_model(x_truth[0:3], x_truth[6:10])
        z = z + self.rng.normal(0.0, self.stds)
        z[1] = np.arctan2(np.sin(z[1]), np.cos(z[1]))   # keep az in [-pi, pi]
        return z

    
class GyroSensor:
    def __init__(self, arw, rrw, bias0_std, dt, seed=101):
        """arw: rate noise sigma_rc [rad/sqrt(s)] (angle random walk), eqn 124
        rrw: bias random walk sigma_wc [rad/sqrt(s^3)], eqn 128
        bias0_std: turn-on bias 1-sigma [rad/s] """
        self.rng = np.random.default_rng(seed)
        self.sigma_rd = arw / np.sqrt(dt)   # eqn 129 (d = discrete; c=continuous)
        self.sigma_wd = rrw * np.sqrt(dt)   # eqn 130
        self.bias = self.rng.normal(0.0, bias0_std, 3)  # true (w/o rnd walk) bias; NEES reads this

    def measure(self, x_truth):
        """measured biasw_m = w + b + n_r, w/ b doing a random walk (eqn 111, 114)."""
        self.bias = self.bias + self.rng.normal(0.0, self.sigma_wd, 3)
        return x_truth[10:13] + self.bias + self.rng.normal(0.0, self.sigma_rd, 3)


# TODO: define sigma
class StarTracker:
    def __init__(self, sigma_cross, sigma_bore, rate_hz, dt, seed=202):
        self.sigmas = np.array([sigma_cross, sigma_cross, sigma_bore])  # boresight = body z
        self.period = max(1, round(1.0 / (rate_hz * dt)))
        self.rng = np.random.default_rng(seed)
        self.k = 0

    def measure(self, x_truth):
        """Noisy attitude quaternion, or None if blinded"""
        self.k += 1
        if self.k % self.period:
            return None
        dtheta = self.rng.normal(0.0, self.sigmas)        # anisotropic(?), sensor frame
        dq = np.array([0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2], 1.0])
        dq = quat_normalize(dq)
        return quat_normalize(quat_multiply(dq, x_truth[6:10]))


class SensorSuite:
    """Gyro (every step) + star tracker + lidar. Pass in sensors,
    each with its own seed. Star tracker / lidar can be wrapped for dropout,
    e.g. lidar=DropoutSensor(LidarSensor(...), windows, dt)"""
    def __init__(self, gyro, star, lidar):
        self.gyro = gyro
        self.star = star
        self.lidar = lidar

    def measure(self, x_truth):
        return {"gyro": self.gyro.measure(x_truth),
                "star": self.star.measure(x_truth),
                "lidar": self.lidar.measure(x_truth)}