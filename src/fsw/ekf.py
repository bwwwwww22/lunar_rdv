"""Multiplicative Extended Kalman Filter (MEKF) for 6-DOF navigation

Filter state x (13):   [r(3), v(3), q(4), b(3)]     (b = gyro bias)
Error state δx (12):   [δr(3), δv(3), δθ(3), Δb(3)]

Covariance P is 12x12. Attitude is propagated with the gyro reading
(w_hat = w_m - b_hat) instead of Euler's equation (last version), 
and the gyro bias is estimated.
Star tracker and lidar are fused with sequential updates 
Attitude error is a 3-element small-angle vector 
Corrections are injected multiplicatively into q_hat

following https://web.stanford.edu/group/arl/sites/default/files/public/publications/aas_13_364.pdf
"""

import numpy as np
from src.utils.quaternions import (quat_multiply, quat_normalize, quat_derivative,
                                   quat_error, quat_to_rotmat)
from src.environment.dynamics import cw_acceleration
from src.fsw.control import cw_state_space
from src.fsw.sensors import lidar_model

import warnings
# Apple Accelerate BLAS can flag matmul with zero rows/cols. I've tested & confirmed 
# that numerical outputs are correct, so I suppressed the warning locally
warnings.filterwarnings(
    "ignore",
    message=".*(divide by zero|overflow|invalid value) encountered in matmul.*",
    category=RuntimeWarning,
)

class MEKF:
    def __init__(self, params, x0, P0, Q, R):
        """
        params : dynamics params (mass, orbit_rate)
        x0     : initial filter state [r, v, q, b] (13,)
        P0     : initial error-state covariance (12,12)
        Q      : continuous process noise (12,12);
                 attitude block = gyro rate noise sigma_r^2
                 last block = bias random walk sigma_w^2
        R      : per-sensor measurement covariances {"star": (3,3), "lidar": (3,3)}
        """
        self.params = params
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)   # how uncertain we are after fusing zk
        self.Q = np.array(Q, dtype=float)
        self.R = {name: np.array(Rk, dtype=float) for name, Rk in R.items()}
        self.w_m = np.zeros(3)               # last gyro reading

    # inject an error-state correction into full state
    def _inject(self, dx):
        """Apply a 12-dim error-state correction dx into the 13-dim filter state.
        Additive for r, v, b; multiplicative for attitude."""
        self.x[0:3]   += dx[0:3]     # position: additive; eqn 61
        self.x[3:6]   += dx[3:6]     # velocity: additive
        self.x[10:13] += dx[9:12]    # gyro bias: additive; b_hat += Δb

        # Attitude: multiplicative. dq ≈ [δθ/2, 1] (eqns 31 & 32)
        dtheta = dx[6:9]
        dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
        dq = quat_normalize(dq)

        # q̂ = dq ⊗ q̂ (eqn 29)
        self.x[6:10] = quat_normalize(quat_multiply(dq, self.x[6:10]))

    def get_estimate(self):
        """[r, v, q, ω_hat], same layout as the truth state, for guidance/control."""
        x = self.x.copy()
        x[10:13] = self.w_m - self.x[10:13]     # T&R eqn 136
        return x

    def get_filter_state(self):
        """Internal [r, v, q, b_hat], for NEES."""
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()


    def predict(self, u, omega_m, dt):
        """Propagate state estimate and error-state covariance
        u       : applied control (6,); only the force drives the filter
        omega_m : gyro reading (3,); drives the attitude kinematics
        State propagation: RK4 (CW + force for translation, gyro for attitude)
        Covariance propagation: Pk⁻ = Fd P Fdᵀ + Q_d, where F is the
        analytic continuous error-state matrix"""

        # diagnostics
        #print(f"predict called, ||P||={np.max(np.abs(self.P)):.3e}, "
        #  f"finite={np.all(np.isfinite(self.P))}")
        #F_c = self._error_state_jacobian(self.x, u)
        #print(f"  F_c max entry: {np.max(np.abs(F_c)):.3e}")

        if not np.all(np.isfinite(self.x)):
            raise RuntimeError(f"x_hat non-finite entering predict: {self.x}")
    
        self.w_m = np.asarray(omega_m, dtype=float)
        w_hat = self.w_m - self.x[10:13]

        # 1. Propagate the filter state (gyro replaces Euler's equation for attitude)
        self.x = _rk4_step_filter(self.x, u, w_hat, dt, self.params)

        # 2. Compute (continuous) error-state Jacobian F_c at current estimate
        F_c = self._error_state_jacobian(w_hat)
        if np.max(np.abs(F_c)) > 1e3:
            print(f"WARNING: large F_c entries, max={np.max(np.abs(F_c)):.3e}")

        # 3. Discretize: F_d ≈ I + F_c * dt  (assuming fast control rates)
        # eqn 54 (1st order truncation from 53)
        F_d = np.eye(12) + F_c * dt
        # also equiv to T&R eqn 164 w/o the 2nd order term (F_c^2 dt^2 / 2)

        # 4. Propagate covariance
        self.P = F_d @ self.P @ F_d.T + self.Q * dt   # Q scaled by dt (continuous->discrete)

        # 5. Enforce symmetry (numerical hygiene)
        self.P = 0.5 * (self.P + self.P.T)

        # Clamp any tiny negative eigenvalues from floating-point drift
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 0:
            self.P += (-min_eig + 1e-12) * np.eye(12)

    def _error_state_jacobian(self, w_hat):
        """Continuous error-state matrix F (12x12):
        translation: CW (no change from last version)
        attitude:    δθ_dot = -[ω̂×]δθ - Δb   (Trawny & Roumeliotis sec 2.4)
        bias:        Δb_dot -> 0 (p.s. T&R eqn says Δb_dot = n_w, but this is the
                     deterministic part of Fc, see eqns 158-160)"""
        A_cw, _ = cw_state_space(self.params["orbit_rate"], self.params["mass"])
        F = np.zeros((12, 12))
        F[0:6, 0:6] = A_cw
        F[6:9, 6:9] = -_skew(w_hat)     # T&R eqn 160
        F[6:9, 9:12] = -np.eye(3)       # T&R eqn 160
        return F

    def _star_residual(self, z):
        """Assume y ≈ δθ + noise for star traker, therefore H = [0 0 I 0]"""
        y = quat_error(z, self.x[6:10])
        H = np.zeros((3, 12))
        H[:, 6:9] = np.eye(3)
        return y, H

    def _lidar_residual(self, z):
        """Lidar [rho, az, el] in the body frame:
        p = C(q_hat)(-r_hat)
        dp/dr = -C(q_hat), dp/dδθ = [px] C(q) ≈ (I - [δθ×]) C(q_hat)
        C(q) is the reference-to-body rotation, i.e. quat_to_rotmat(...)

        T&R sec 3.1:
        eqn 202-203: q = δq ⊗ q_hat gives C(q) = C(δq)·C(q_hat)
        eqn 204: C(δq) ≈ I - ⌊δθ×⌋
        eqn 206: C(q) - C(q_hat) = -⌊δθx⌋·C(q_hat)
        eqn 208-209: ⌊C(q_hat)·r ×⌋ δθ, here we use -r instead bc that's
            line of sight from chaser to tgt. we can rewrite as ⌊px⌋δθ
        eqn 210-211: stacking w/ a zero bias coln

        In T&R their filter has no pos state, and their sun-sensor is a
        known constant, so eqn 211 has no position column at all. 
        Her we have measurement h = f(p), where f is the conversion to
        [rho, az, el]. Differentiate wrt r and hold q_hat fixed:
        dh/dr = (df/dp)·(dp/dr) = J·-C(q_hat)

        """
        z_hat, p = lidar_model(self.x[0:3], self.x[6:10])   # input: r, q
        y = z - z_hat
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))     # wrap az residual
        # J acts like projection matrix Π (in T&R), except it's a jacobian
        J = _spherical_jacobian(p)
        H = np.zeros((3, 12))
        # like in T&R eqn 211, col 9:12 = 0, but 6:9 is like the T&R sun-senor example
        H[:, 6:9] = J @ _skew(p)
        H[:, 0:3] = J @ (-quat_to_rotmat(self.x[6:10]))
        return y, H

    # TODO: compare with old
    def update(self, z_dict):
        """Sequentially fuse each available measurement in z_dict ("star", "lidar").
        The gyro is used in predict, not here.
        Sensors that returned None are skipped.
        Returns [(name, y, S), ...] for NIS"""

        residuals = {"star": self._star_residual, "lidar": self._lidar_residual}
        results = []
        for name, residual in residuals.items():
            z = z_dict.get(name)
            if z is None:
                continue
            if not np.all(np.isfinite(self.P)):
                raise RuntimeError(
                    f"P non-finite after step. "
                    f"max|P|={np.nanmax(np.abs(self.P))}, "
                    f"min_eig={np.min(np.linalg.eigvalsh(np.nan_to_num(self.P)))}"
                )
            if np.max(np.abs(self.P)) > 1e12:
                raise RuntimeError(
                    f"P exploded to max|P|={np.max(np.abs(self.P)):.3e} — "
                    f"filter is diverging"
                )
            
            # Innovation (3,) and measurement Jacobian (3,12)
            # Eval at the current estimate, which the previous sensor may have just updated(?)
            # Noise is still additive (J = I), so S = HPHᵀ + R
            y, H = residual(z)
            R = self.R[name]
            S = H @ self.P @ H.T + R                       # (3,3); eqn 58

            # Kalman gain K = P Hᵀ S⁻¹ (parallel to eqn 58)
            K = np.linalg.solve(S, H @ self.P).T           # (12,3)

            # Error-state correction (eqn 59)
            dx = K @ y                                     # (12,)

            # Inject correction into full state (multiplicative for attitude)
            self._inject(dx)

            # Covariance update; Joseph form for numerical stability (P stays symmetric)
            # https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
            I_KH = np.eye(12) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

            # symmetry
            self.P = 0.5 * (self.P + self.P.T)

            # Clamp any tiny negative eigenvalues from floating-point drift
            min_eig = np.min(np.linalg.eigvalsh(self.P))
            if min_eig < 0: self.P += (-min_eig + 1e-12) * np.eye(12)

            results.append((name, y, S))

        return results

def _rk4_step_filter(x, u, w_hat, dt, params):
    """RK4 propagation of the 13-state filter state over dt, holding ω_hat
    and the applied force constant.
    - Translation: CW + F/m
    - Attitude: quaternion driven by ω_hat (no Euler eqn, so filter needs no inertia)
    - Bias: b_dot = 0 """
    n, m = params["orbit_rate"], params["mass"]
    F = np.asarray(u, dtype=float)[0:3]

    def f(s):   # time derivative of the state
        ds = np.zeros(13)
        ds[0:3] = s[3:6]    # r_dot=v
        ds[3:6] = cw_acceleration(s[0:3], s[3:6], n) + F / m    #v_dot=a
        ds[6:10] = quat_derivative(s[6:10], w_hat)      #q_dot
        # bias is constant, so b_dot = 0
        return ds

    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    x_new = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x_new[6:10] /= np.linalg.norm(x_new[6:10])
    return x_new


def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])

def _spherical_jacobian(p):
    """jacobian of cartesian-to-spherical coordinates.
    d[rho, az, el]/dp for a line-of-sight vector p """
    x, y, z = p
    rho2 = x*x + y*y + z*z
    rxy2 = x*x + y*y
    rho, rxy = np.sqrt(rho2), np.sqrt(rxy2)
    return np.array([
        [x/rho,             y/rho,             z/rho],
        [-y/rxy2,           x/rxy2,            0.0],
        [-x*z/(rho2*rxy),   -y*z/(rho2*rxy),   rxy/rho2],
    ])