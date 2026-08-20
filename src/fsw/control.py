"""LQR flight controller. Designs two decoupled LQR gains (translation +
attitude) about the docking setpoint, then computes u = -K (x - x_ref).

Translation (Clohessy-Wiltshire) is linear.
Attitude model is linearized about the target attitude (small-angle)."""

import numpy as np
from scipy.linalg import solve_continuous_are
from src.utils.quaternions import quat_error

def cw_state_space(n, mass):
    """Continuous-time CW translational dynamics (Hill frame).
    ẋ = A x + B u
    ẋ = [xdot, ydot, zdot, vxdot, vydot, vzdot]
    x = [x, y, z, vx, vy, vz] (state)
    u = [Fx, Fy, Fz] (input)
    A = (6,6), B = (6,3)
    """
    A = np.zeros((6, 6))
    # position derivatives = velocity
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    # velocity derivatives = accelerations
    A[3, 0] = 3 * n**2      # ax from x
    A[3, 4] = 2 * n         # ax from vy
    A[4, 3] = -2 * n        # ay from vx
    A[5, 2] = -n**2         # az from z

    B = np.zeros((6, 3))
    B[3, 0] = 1.0 / mass
    B[4, 1] = 1.0 / mass
    B[5, 2] = 1.0 / mass
    return A, B


def attitude_state_space(inertia):
    """Continuous-time attitude dynamics linearized about the target attitude
    (small-angle error, which should be fine for our docking scenario) 
    with zero nominal angular rate.

    ẋ = A x + B u
    ẋ = [dtheta_dot, w_dot]
    x = [dtheta_x, dtheta_y, dtheta_z, wx, wy, wz] (state)
        dtheta = small-angle attitude error (rad)
        w = angular velocity (rad/s)
    u = [taux, tauy, tauz] (input)
    A = (6,6), B = (6,3)

    Gyroscopic term w×Iw drops out when w~0, so the dynamics are linear:
    I·w_dot ≈ tau, and dθ̇ ≈ w.
    """
    I_inv = np.linalg.inv(inertia)

    A = np.zeros((6, 6))
    A[0, 3] = 1.0   # dtheta_x_dot = wx
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    # w_dot = I_inv @ tau  (no state feedback in A near zero rate)

    B = np.zeros((6, 3))
    B[3:6, :] = I_inv
    return A, B


def lqr_gain(A, B, Q, R):
    """Continuous LQR. Returns gain K where u = -K x."""
    P = solve_continuous_are(A, B, Q, R)    # solving CARE: A^T P + P A - P B R^-1 B^T P + Q = 0
    K = np.linalg.inv(R) @ B.T @ P  # K = R⁻¹ Bᵀ P
    return K


# ---------- Controller ----------

class LQRController:
    def __init__(self, params, Q_trans, R_trans, Q_att, R_att):
        n = params["orbit_rate"]
        mass = params["mass"]
        inertia = params["inertia"]

        A_t, B_t = cw_state_space(n, mass)
        A_a, B_a = attitude_state_space(inertia)

        # builds both gains
        self.K_trans = lqr_gain(A_t, B_t, Q_trans, R_trans)   # (3,6)
        self.K_att = lqr_gain(A_a, B_a, Q_att, R_att)         # (3,6)

    def command(self, x_est, x_ref):
        """Compute control u(6,) = [F(3), tau(3)] from est and ref.
        x_est, x_ref: (13,) = [pos(3), vel(3), quat(4), w(3)] in Hill
        """
        # --- Translation error ---
        pos_err = x_est[0:3] - x_ref[0:3]
        vel_err = x_est[3:6] - x_ref[3:6]
        e_trans = np.concatenate([pos_err, vel_err])
        F = -self.K_trans @ e_trans

        # --- Attitude error (small-angle vector, NOT quaternion subtraction!) ---
        q_est = x_est[6:10]
        q_ref = x_ref[6:10]
        dtheta = quat_error(q_est, q_ref)      # (3,) small-angle error; q_err = q_ref * q_est^-1
        w_err = x_est[10:13] - x_ref[10:13]
        e_att = np.concatenate([dtheta, w_err])
        tau = -self.K_att @ e_att

        return np.concatenate([F, tau])
