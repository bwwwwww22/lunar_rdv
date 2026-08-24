"""Quaternion operations. Convention: scalar-last [x, y, z, w], JPL.
Unit quaternions represent a PASSIVE transform from the reference
frame to the chaser body frame: v_body = R(q) @ v_ref."""

import numpy as np


def quat_multiply(q1, q2):
    """product q1 ⊗ q2. Composition matches DCM chaining:
    R(q1 ⊗ q2) = R(q1) @ R(q2)."""
    v1, w1 = q1[0:3], q1[3]
    v2, w2 = q2[0:3], q2[3]
    v = w1*v2 + w2*v1 - np.cross(v1, v2)
    w = w1*w2 - np.dot(v1, v2)
    return np.array([v[0], v[1], v[2], w])


def quat_normalize(q):
    """Renormalize to unit length. Call after every integration step."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion.")
    q_norm = q / n
    return q_norm


def quat_conjugate(q):
    """Inverse for unit quaternions: negate vector part only."""
    x, y, z, w = q
    q_inv = np.array([-x, -y, -z, w])
    return q_inv


def quat_to_rotmat(q):
    """Rotation matrix such that v_body = R @ v_ref (passive, ref->body)"""
    x, y, z, w = quat_normalize(q)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y + w*z),     2*(x*z - w*y)],
        [2*(x*y - w*z),     1 - 2*(x*x + z*z),     2*(y*z + w*x)],
        [2*(x*z + w*y),         2*(y*z - w*x), 1 - 2*(x*x + y*y)],
    ])
    return R


def quat_derivative(q, omega):
    """q̇ = 0.5 * [w, 0] ⊗ q (LEFT-multiplication). 
    w is body-frame angular velocity (rad/s)."""
    omega_quat = np.array([omega[0], omega[1], omega[2], 0.0])
    q_dot = 0.5 * quat_multiply(omega_quat, q)
    return q_dot


def quat_error(q_est, q_ref):
    """Attitude error as a small-angle 3-vector (rad)
    Returns the vector part of (q_est ⊗ q_ref^-1), scaled by 2 for small angles."""
    q_err = quat_multiply(q_est, quat_conjugate(q_ref))
    if q_err[3] < 0:            # shortest rotation
        q_err = -q_err
    return 2.0 * q_err[0:3]     # ~ [roll, pitch, yaw] error for small angles


def quat_to_euler(q):
    """3-2-1 intrinsic (roll-pitch-yaw, body-fixed axes) Euler angles (rad)"""
    x, y, z, w = quat_normalize(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0)) # clip to avoid sinp drifting
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])
