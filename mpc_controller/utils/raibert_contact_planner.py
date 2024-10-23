import numpy as np
import pinocchio as pin

from ..config.config_abstract import GaitConfig

class RaiberContactPlanner():
    GRAVITY = 9.81
    VEL_TRACKING_COEFF = 0.05
    """
    Raiber contact planner class.
    Plans contact locations for locomotion on flat terrain.
    """
    def __init__(self,
                 offset_hip_b : np.ndarray,
                 config_gait : GaitConfig, 
                 height_offset : float = 0.) -> None:

        self.config_gait = config_gait
        self.height_offset = height_offset

        cnt_offset = 0.1
        offset_hip_b[:, 1] += np.array([cnt_offset, -cnt_offset, cnt_offset, -cnt_offset])
        
        self.offset_hip_b = offset_hip_b
        self.N_feet = len(self.offset_hip_b)

    def set_config_gait(self, config_gait = GaitConfig) -> None:
        self.config_gait = config_gait
    
    def next_contact_locations(self,
                               q : np.ndarray,
                               com_xyz : np.ndarray,
                               time_to_cnt : np.ndarray,
                               v_des : np.ndarray = np.zeros(3),
                               w_yaw : float = 0.,
                              ):
        
        assert len(time_to_cnt) == self.N_feet, f"time_to_cnt should be of length {self.N_feet}."
        com_xy, com_z = com_xyz[:2], com_xyz[-1] - self.height_offset

        # R base in world frame
        w_T_b = pin.XYZQUATToSE3(q[:7])

        # To world frame
        v_des_w = w_T_b @ v_des
        v_des_w[2] = 0.
        v_des = w_T_b.T @ v_des_w
        vtrack = v_des[:2]
        
        # Set roll and pitch to 0.
        rpy_vector = pin.rpy.matrixToRpy(w_T_b.rotation)
        rpy_vector[:2] = 0.0  # Set roll and pitch to 0
        R_horizontal = pin.rpy.rpyToMatrix(rpy_vector)

        cnt_pos_w = np.zeros((self.N_feet, 3))
        cnt_pos_w[:, -1] = self.height_offset

        for foot_id, (offset_hip, t_to_cnt, stance_ratio) in enumerate(zip(
            self.offset_hip_b,
            time_to_cnt,
            self.config_gait.stance_ratio
            )):

            if t_to_cnt > 0:
                hip_loc = com_xy + (R_horizontal @ offset_hip)[0:2] + t_to_cnt * vtrack
                raibert_step = (0.5 * vtrack * self.config_gait.nominal_period * stance_ratio)
                ang_step = np.cross(0.5 * np.sqrt(com_z / RaiberContactPlanner.GRAVITY) * vtrack, [0.0, 0.0, w_yaw])
            
                cnt_pos_w[foot_id][:2] = raibert_step + hip_loc + ang_step[0:2] + q[:2]
            else:
                cnt_pos_w[foot_id][-1] = -1.
            
        return cnt_pos_w
    
    def next_contact_location(self,
                            i_foot : int,
                            q : np.ndarray,
                            com_xyz : np.ndarray,
                            time_to_cnt : np.ndarray,
                            v_des : np.ndarray = np.zeros(3),
                            w_yaw : float = 0.,
                            ):
        
        com_xy, com_z = com_xyz[:2], com_xyz[-1] - self.height_offset
        vtrack = v_des[:2]

        # R base in world frame
        w_T_b = pin.XYZQUATToSE3(q[:7])
        
        # Set roll and pitch to 0.
        rpy_vector = pin.rpy.matrixToRpy(w_T_b.rotation)
        rpy_vector[:2] = 0.0  # Set roll and pitch to 0
        R_horizontal = pin.rpy.rpyToMatrix(rpy_vector)

        cnt_pos_w = np.zeros(3)
        cnt_pos_w[-1] = self.height_offset

        if time_to_cnt > 0:
            hip_loc = com_xy + (R_horizontal @ self.offset_hip_b[i_foot])[0:2] + time_to_cnt * vtrack
            raibert_step = (0.5 * vtrack * self.config_gait.nominal_period * self.config_gait.stance_ratio[i_foot])
            ang_step = np.cross(0.5 * np.sqrt(com_z / RaiberContactPlanner.GRAVITY) * vtrack, [0.0, 0.0, w_yaw])
        
            cnt_pos_w[:2] = raibert_step + hip_loc + ang_step[0:2]
        else:
            cnt_pos_w[-1] = -1.
            
        return cnt_pos_w