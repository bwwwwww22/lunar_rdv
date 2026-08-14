"""Quaternion operations. Convention: scalar-first [w, x, y, z], Hamilton product,
unit quaternions represent the attitude of the chaser body relative to reference."""

import numpy as np


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    prod = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return prod


def quat_normalize(q):
    """Renormalize to unit length. Call after every integration step."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    q_norm = q / n
    return q_norm


def quat_conjugate(q):
    """Inverse for unit quaternions. negate vector part?"""
    w, x, y, z = q
    q_inv = np.array([w, -x, -y, -z])
    return q_inv


def quat_to_rotmat(q):
    """Rotation matrix such that v_ref = R @ v_body (for body->ref)"""
    w, x, y, z = quat_normalize(q)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return R


def quat_derivative(q, omega):
    """q̇ = 0.5 * q ⊗ [0, w]. w is body-frame angular velocity (rad/s)"""
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * quat_multiply(q, omega_quat)
    return q_dot


def quat_error(q_est, q_ref):
    """Attitude error as a small-angle 3-vector (rad), for LQR state error.
    Returns the vector part of (q_ref^-1 ⊗ q_est), scaled by 2 for small angles."""
    q_err = quat_multiply(quat_conjugate(q_ref), q_est)
    if q_err[0] < 0:            # shortest rotation
        q_err = -q_err
    return 2.0 * q_err[1:]      # ~ [roll, pitch, yaw] error for small angles


def quat_to_euler(q):
    """yaw-pitch-roll Euler angles (rad)"""
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0)) # clip to avoid sinp drifting
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])
