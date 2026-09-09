"""Multiplicative Extended Kalman Filter (MEKF) for 6-DOF navigation

Full state estimate x_hat (13):  [r(3), v(3), q(4), ω(3)]
Error state δx (12):          [δr(3), δv(3), δθ(3), δω(3)]

Covariance P is 12x12. Attitude error is a 3-element small-angle vector;
corrections are injected multiplicatively into q_hat.

following https://web.stanford.edu/group/arl/sites/default/files/public/publications/aas_13_364.pdf
"""

import numpy as np
from src.utils.quaternions import quat_multiply, quat_normalize
#from src.environment.dynamics import state_derivative

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
        params : dynamics params (mass, inertia, inertia_inv, orbit_rate)
        x0     : initial full-state estimate (13,)
        P0     : initial error-state covariance (12,12)
        Q      : process noise covariance (12,12)
        R      : measurement noise covariance (matches measurement dim)
        """
        self.params = params
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)   # how uncertain we are after fusing zk
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

    # inject an error-state correction into full state
    def _inject(self, dx):
        """Apply a 12-dim error-state correction dx into the 13-dim full state.
        Additive for r, v, ω; multiplicative for attitude."""
        self.x[0:3]   += dx[0:3]     # position: additive; eqn 61
        self.x[3:6]   += dx[3:6]     # velocity: additive
        self.x[10:13] += dx[9:12]     # angular rate: additive

        # Attitude: multiplicative. dq ≈ [δθ/2, 1] (eqns 31 & 32)
        dtheta = dx[6:9]
        dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
        dq = quat_normalize(dq)

        # q̂ = dq ⊗ q̂ (eqn 29)
        self.x[6:10] = quat_normalize(quat_multiply(dq, self.x[6:10]))

    def get_estimate(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()

    # ---------- Predict step ----------
    def predict(self, u, dt):
        """Propagate state estimate and error-state covariance
        State propagation: RK4
        Covariance propagation: Pk⁻ = Fd P Fdᵀ + Q_d, where F is the discrete
        error-state transition matrix computed from a numerical Jacobian of
        the continuous dynamics."""

        # diagnostics
        #print(f"predict called, ||P||={np.max(np.abs(self.P)):.3e}, "
        #  f"finite={np.all(np.isfinite(self.P))}")

        #F_c = self._error_state_jacobian(self.x, u)  # compute EARLY to inspect
        #print(f"  F_c max entry: {np.max(np.abs(F_c)):.3e}")

        if not np.all(np.isfinite(self.x)):
            raise RuntimeError(f"x_hat non-finite entering predict: {self.x}")
    
        # 1. Propagate the full-state estimate via the same RK4 as plant
        self.x = _rk4_step_estimate(self.x, u, dt, self.params)

        # 2. Compute (continuous) error-state Jacobian F_c at current estimate
        F_c = self._error_state_jacobian(self.x, u)
        if np.max(np.abs(F_c)) > 1e3:
            print(f"WARNING: large F_c entries, max={np.max(np.abs(F_c)):.3e}")

        # 3. Discretize: F_d ≈ I + F_c * dt  (assuming fast control rates)
        # eqn 54 (1st order truncation from 53)
        F_d = np.eye(12) + F_c * dt

        # 4. Propagate covariance
        self.P = F_d @ self.P @ F_d.T + self.Q * dt   # Q scaled by dt (continuous->discrete)

        # 5. Enforce symmetry (numerical hygiene)
        self.P = 0.5 * (self.P + self.P.T)

        # Clamp any tiny negative eigenvalues from floating-point drift
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 0:
            self.P += (-min_eig + 1e-12) * np.eye(12)

        '''
        # diagnostic (first 3 steps)
        self._n_predicts = getattr(self, "_n_predicts", 0) + 1
        verbose = self._n_predicts <= 3
        if verbose:
            print(f"\n=== predict {self._n_predicts} ===")
            print(f"  ||P|| in:  {np.max(np.abs(self.P)):.3e}, finite={np.all(np.isfinite(self.P))}")
            print(f"  ||x|| in:  {np.max(np.abs(self.x)):.3e}")
            print(f"  ||F_d||:   {np.max(np.abs(F_d)):.3e}")
            print(f"  ||Q*dt||:  {np.max(np.abs(self.Q * dt)):.3e}")
            FPF = F_d @ self.P @ F_d.T
            print(f"  ||FPF||:   {np.max(np.abs(FPF)):.3e}, finite={np.all(np.isfinite(FPF))}")
            print(f"  ||P|| out: {np.max(np.abs(self.P)):.3e}, finite={np.all(np.isfinite(self.P))}")
        '''


    def _error_state_jacobian(self, x, u, eps=1e-4):
        """Numerical Jacobian of the error-state dynamics about the current
        estimate. Uses central differences on the full state, then
        maps quaternion perturbations to small-angle δθ perturbations.
        For each error-state axis i:
        inject a small perturbation, evaluate f at the perturbed state, 
        compute the error-state derivative and get central-difference """

        F = np.zeros((12, 12))
        for i in range(12):
            x_plus  = _perturb_full_state(x, i,  eps)
            x_minus = _perturb_full_state(x, i, -eps)
            f_plus  = _error_state_derivative(x_plus,  u, self.params)
            f_minus = _error_state_derivative(x_minus, u, self.params)
            F[:, i] = (f_plus - f_minus) / (2 * eps)

        return F

    def _measurement_residual(self, z, x_hat):
        """Compute the 12-dim innovation between measurement z (13,) and
        current estimate x_hat (13,). Position/velocity/rate: additive.
        Attitude: small-angle error from quaternion residual."""
        from src.utils.quaternions import quat_error
        y = np.zeros(12)
        y[0:3] = z[0:3]   - x_hat[0:3]                 # position residual
        y[3:6] = z[3:6]   - x_hat[3:6]                 # velocity residual
        y[6:9] = quat_error(z[6:10], x_hat[6:10])      # attitude residual (δθ)
        y[9:12] = z[10:13] - x_hat[10:13]              # angular-rate residual
        return y

    def update(self, z):
        """Fuse a full-state pose measurement z (13,) into the estimate.
        Measure the whole state directly (with the synthetic PoseSensor). 
        Measurement residual for attitude is computed as a small-angle 
        quaternion error so it lives in the same 3-DOF space as the covariance."""

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
        
        # Innovation (measurement - prediction) in ERROR-STATE space (12)
        y = self._measurement_residual(z, self.x)      # (12,)

        # Innovation covariance is S = HPHᵀ + JRJᵀ, but 2 major assumptions made S=P+R
        # 1. synthetic pose sensor measures the 12-D error state directly (measurement error = state error)
        #   H is partial derivative of predicted measurement wrt the error state = identity here
        # 2. measurement noise is additive in the error-state space
        #   J is partial derivative of the measurement equation wrt the measurement noise vector = identity here
        H = np.eye(12)      # replaces eqn 63
        S = H @ self.P @ H.T + self.R                  # (12,12); eqn 58

        # Kalman gain (parallel to eqn 58)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(12))   # (12,12)

        # Error-state correction (eqn 59)
        dx = K @ y                                     # (12,)

        # Inject correction into full state (multiplicative for attitude)
        self._inject(dx)

        # Covariance update; Joseph form for numerical stability (P stays symmetric)
        # https://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
        I_KH = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Clamp any tiny negative eigenvalues from floating-point drift
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 0: self.P += (-min_eig + 1e-12) * np.eye(12)

        return y, S


def _rk4_step_estimate(x, u, dt, params):
    """RK4 propagation of the full 13-state estimate using the same dynamics
    as the plant. Renormalizes quaternion at the end."""
    from src.environment.dynamics import state_derivative
    k1 = state_derivative(x, u, params)
    k2 = state_derivative(x + 0.5*dt*k1, u, params)
    k3 = state_derivative(x + 0.5*dt*k2, u, params)
    k4 = state_derivative(x + dt*k3, u, params)
    x_new = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    x_new[6:10] /= np.linalg.norm(x_new[6:10])
    return x_new


def _error_state_derivative(x, u, params):
    """Compute the 12-dim error-state derivative from the 13-dim full-state
    derivative (from dynamics.state_derivative). Positions/velocities/rates 
    map directly; the quaternion derivative maps to a small-angle rate via
    q̇ = ½ [ω, 0] ⊗ q  →  δθ̇ ≈ ω."""
    from src.environment.dynamics import state_derivative
    xdot = state_derivative(x, u, params)
    dx = np.zeros(12)
    dx[0:3]  = xdot[0:3]     # ṙ = v
    dx[3:6]  = xdot[3:6]     # v̇ (from CW + control)
    dx[6:9]  = x[10:13]      # δθ̇ ≈ ω (body-frame angular velocity)
    dx[9:12] = xdot[10:13]   # ω̇ (from Euler's equation)
    return dx


def _perturb_full_state(x, i, eps):
    """Apply a small perturbation to error-state axis i, returning the
    perturbed full state. For attitude (indices 6:9), the perturbation
    is applied multiplicatively as a small-angle rotation."""
    xp = x.copy()
    if i < 6:                     # position or velocity: additive
        xp[i] += eps
    elif i < 9:                   # attitude: multiplicative small rotation
        dtheta = np.zeros(3); dtheta[i - 6] = eps
        dq = np.array([0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2], 1.0])
        dq /= np.linalg.norm(dq)
        xp[6:10] = quat_multiply(dq, xp[6:10])
        xp[6:10] /= np.linalg.norm(xp[6:10])
    else:                         # angular rate: additive
        xp[10 + (i - 9)] += eps   # shift index range from 9:12 to 10:13
    return xp