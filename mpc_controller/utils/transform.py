import pinocchio as pin
import numpy as np

def quat_to_ypr_state(q_full: np.ndarray):
    # Convert quaternion to rotation matrix
    R = pin.Quaternion(q_full[6], *q_full[3:6]).toRotationMatrix()
    
    # Convert rotation matrix to RPY
    rpy_angles = pin.rpy.matrixToRpy(R)

    # Combine the position, YPR angles, and other states
    q_euler = np.hstack([q_full[:3], rpy_angles[[2, 1, 0]], q_full[7:]])

    return q_euler

def ypr_to_quat_state(q_euler: np.ndarray):
    # Extract YPR angles
    ypr_angles = q_euler[3:6]

    # Convert RPY to rotation matrix
    R = pin.rpy.rpyToMatrix(ypr_angles[[2, 1, 0]])

    # Combine the position, quaternion, and other states
    q_full = np.hstack([q_euler[:3], pin.Quaternion(R).coeffs(), q_euler[6:]])

    return q_full

def local_angular_to_euler_derivative(q_euler: np.ndarray, w_local: np.ndarray):
    cx = np.cos(q_euler[2])
    sx = np.sin(q_euler[2])
    cy = np.cos(q_euler[1])
    sy = np.sin(q_euler[1])
    transform_ = np.array([[0, sx / cy, cx / cy], [0, cx, -sx], [1, sx * sy / cy, cx * sy / cy]])
    return transform_ @ w_local.reshape(-1, 1);

def euler_derivative_to_local_angular(q_euler: np.ndarray, v_euler : np.ndarray):
    cx = np.cos(q_euler[2])
    sx = np.sin(q_euler[2])
    cy = np.cos(q_euler[1])
    sy = np.sin(q_euler[1])
    transform_ = np.array([[-sy, 0, 1], [cy * sx, cx, 0], [cx * cy, -sx, 0]])
    return transform_ @ v_euler.reshape(-1, 1);
