import pinocchio as pin
import numpy as np

def quat_to_ypr_state(q_full: np.ndarray) -> np.ndarray:
    # Convert quaternion to rotation matrix
    R = pin.Quaternion(q_full[3:7]).toRotationMatrix()
    
    # Convert rotation matrix to RPY
    rpy_angles = pin.rpy.matrixToRpy(R)

    # Combine the position, YPR angles, and other states
    q_euler = np.hstack([q_full[:3], rpy_angles[::-1], q_full[7:]])

    return q_euler

def ypr_to_quat_state(q_euler: np.ndarray) -> np.ndarray:
    # Extract YPR angles
    ypr_angles = q_euler[3:6]

    # Convert RPY to rotation matrix
    R = pin.rpy.rpyToMatrix(ypr_angles[::-1])

    # Combine the position, quaternion, and other states
    q_full = np.hstack([q_euler[:3], pin.Quaternion(R).coeffs(), q_euler[6:]])

    return q_full

def ypr_to_quat_state_batched(q_euler_batch: np.ndarray) -> np.ndarray:
    """
    Perform ypr euler to quaternion for several states
    q_euler_batch (shape [N, 18]).
    """
    quat_batched = np.array(
        [pin.Quaternion(pin.rpy.rpyToMatrix(ypr[::-1])).coeffs()
        for ypr
        in q_euler_batch[:, 3:6]]
        )

    q_full_batched = np.hstack(
        (q_euler_batch[:, :3], quat_batched, q_euler_batch[:, 6:])
    )

    return q_full_batched

def v_global_linear_to_local_linear(q_euler : np.ndarray, v_glob_linear : np.ndarray) -> np.ndarray:
    """
    Transform a velocity state from global linear velocity to local angular vel.
    Euler -> pinocchio Freeflyer
    """
    R_BW = pin.rpy.rpyToMatrix(q_euler[3:6][::-1]).T
    v_local_linear = v_glob_linear.copy()
    v_local_linear[3:6] = R_BW @ v_glob_linear[3:6]

    return v_local_linear

def v_local_linear_to_global_linear_batched(q_euler : np.ndarray, v_local_linear : np.ndarray) -> np.ndarray:
    """
    Transform a velocity state from global linear velocity to local angular vel.
    """
    v_glob_linear = v_local_linear.copy()

    w_glob_linear = np.array([
        pin.rpy.rpyToMatrix(ypr[::-1]) @ w_local
        for ypr, w_local
        in zip(q_euler[:, 3:6], v_local_linear[:, 3:6])
    ])

    v_glob_linear[:, 3:6] = w_glob_linear

    return v_glob_linear

def local_angular_to_euler_derivative(ypr_euler: np.ndarray, w_local: np.ndarray) -> np.ndarray:
    cx = np.cos(ypr_euler[2])
    sx = np.sin(ypr_euler[2])
    cy = np.cos(ypr_euler[1])
    sy = np.sin(ypr_euler[1])
    transform_ = np.array([[0, sx / cy, cx / cy], [0, cx, -sx], [1, sx * sy / cy, cx * sy / cy]])
    return transform_ @ w_local.reshape(-1, 1);

def euler_derivative_to_local_angular(ypr_euler: np.ndarray, v_euler : np.ndarray) -> np.ndarray:
    cx = np.cos(ypr_euler[2])
    sx = np.sin(ypr_euler[2])
    cy = np.cos(ypr_euler[1])
    sy = np.sin(ypr_euler[1])
    transform_ = np.array([[-sy, 0, 1], [cy * sx, cx, 0], [cx * cy, -sx, 0]])
    return transform_ @ v_euler.reshape(-1, 1)

def v_to_euler_derivative(q_euler: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_euler_d = np.concatenate((
            pin.rpy.rpyToMatrix(q_euler[3:6][::-1]) @ v[:3],
            local_angular_to_euler_derivative(q_euler[3:6], v[3:6]).flatten(),
            v[6:],
            ))
    return v_euler_d

def v_glob_to_local_batched(q_euler_batch: np.ndarray, v_euler_batch: np.ndarray) -> np.ndarray:
    '''
    Pinocchio Translation + ZYX -> mujoco
    '''
    # Rotate the linear velocity using YPR orientation
    rotated_v_batch = np.array([
        pin.rpy.rpyToMatrix(ypr_euler[::-1]).T @ v_euler
        for ypr_euler, v_euler in zip(q_euler_batch[:, 3:6], v_euler_batch[:, :3])
    ])

    # Convert the angular velocity from global euler to local frame
    v_local_batch = np.array([
        euler_derivative_to_local_angular(ypr_euler, w_euler)
        for ypr_euler, w_euler in zip(q_euler_batch[:, 3:6], v_euler_batch[:, 3:6])
    ]).reshape(-1, 3)

    # Combine the rotated linear velocity and local angular velocity
    v_euler_d = np.hstack((
        rotated_v_batch[:, :3],
        v_local_batch,
        v_euler_batch[:, 6:]
    ))
    
    return v_euler_d