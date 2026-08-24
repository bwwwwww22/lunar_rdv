"""Synthetic sensor model (13-state)"""

import numpy as np

from src.utils.quaternions import quat_multiply, quat_normalize

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