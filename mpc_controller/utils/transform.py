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

def ypr_to_quat_state_batched(q_euler_batch: np.ndarray):
    """
    Perform ypr euler to quaternion for several states
    q_euler_batch (shape [N, 18]).
    """
    quat_batched = np.array(
        [pin.Quaternion(pin.rpy.rpyToMatrix(ypr[[2, 1, 0]])).coeffs()
        for ypr
        in q_euler_batch[:, 3:6]]
        )

    q_full_batched = np.hstack(
        (q_euler_batch[:, :3], quat_batched, q_euler_batch[:, 6:])
    )

    return q_full_batched

def local_angular_to_euler_derivative(ypr_euler: np.ndarray, w_local: np.ndarray):
    cx = np.cos(ypr_euler[2])
    sx = np.sin(ypr_euler[2])
    cy = np.cos(ypr_euler[1])
    sy = np.sin(ypr_euler[1])
    transform_ = np.array([[0, sx / cy, cx / cy], [0, cx, -sx], [1, sx * sy / cy, cx * sy / cy]])
    return transform_ @ w_local.reshape(-1, 1);

def euler_derivative_to_local_angular(ypr_euler: np.ndarray, v_euler : np.ndarray):
    cx = np.cos(ypr_euler[2])
    sx = np.sin(ypr_euler[2])
    cy = np.cos(ypr_euler[1])
    sy = np.sin(ypr_euler[1])
    transform_ = np.array([[-sy, 0, 1], [cy * sx, cx, 0], [cx * cy, -sx, 0]])
    return transform_ @ v_euler.reshape(-1, 1)

def v_to_euler_derivative(q_euler: np.ndarray, v: np.ndarray):
    v_euler_d = np.concatenate((
            v[:3],
            local_angular_to_euler_derivative(q_euler[3:6], v[3:6]).flatten(),
            v[6:],
            ))
    return v_euler_d

def v_to_local_angular_batched(q_euler_batch: np.ndarray, v_euler_batch: np.ndarray):
    v_local_batch = np.array([
        euler_derivative_to_local_angular(q_euler, v_euler) for
        q_euler, v_euler in zip(q_euler_batch[:, 3:6], v_euler_batch[:, 3:6])
    ]).reshape(-1, 3)

    v_euler_d = np.hstack((
        v_euler_batch[:, :3],
        v_local_batch,
        v_euler_batch[:, 6:]
    ))
    return v_euler_d
