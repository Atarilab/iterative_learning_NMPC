import numpy as np
import pinocchio as pin

from ..config.config_abstract import GaitConfig

class RaiberContactPlanner():
    GRAVITY = 9.81
    DEFAULT_FOOT_SIZE = 0.0055 # m
    """
    Raiber contact planner class.
    Plans contact locations for locomotion on flat terrain.
    """
    def __init__(self,
                 offset_hip_b : np.ndarray,
                 config_gait : GaitConfig, 
                 height_offset : float = 0.,
                 x_offset : float = 0.,
                 y_offset : float = 0.,
                 foot_size : float = DEFAULT_FOOT_SIZE,
                 cache_cnt : bool = True,
                 ) -> None:
        """
        Contact planner (locations) based on Raibert heuristic.

        Args:
            offset_hip_b (np.ndarray): Initial offsets of the hips relative to the base 
                in the base frame, shape (N_feet, 3).
            config_gait (GaitConfig): Gait configuration specifying gait parameters such as 
                stance ratio, nominal period, and phase offsets.
            height_offset (float, optional): Vertical offset for the contact points from 
                the ground. Defaults to 0.0.
            x_offset (float, optional): Horizontal offset in the x-direction for the initial 
                hip positions. Adjusts the offsets to account for a preferred forward/backward 
                stance. Defaults to 0.0.
            y_offset (float, optional): Horizontal offset in the y-direction for the initial 
                hip positions. Adjusts the offsets to account for a preferred lateral stance. 
                Defaults to 0.0.
            cache_cnt (bool, optional): If True, caches previously computed contact locations 
                for faster replanning. Avoids changing the contact locations already computed when
                replanning.
                Defaults to True.
        """
        self.config_gait = config_gait
        self.height_offset = height_offset
        self.foot_size = foot_size
        self.cache_cnt = cache_cnt

        self.offset_hip_b = offset_hip_b
        self.offset_hip_b[:, 0] += np.array([x_offset, x_offset, -x_offset, -x_offset])
        self.offset_hip_b[:, 1] += np.array([y_offset, -y_offset, y_offset, -y_offset])

        self.N_feet = len(self.offset_hip_b)
        self.default_cnt_w = np.zeros(3)
        # {foot : {time_cnt : pos}}
        self.planed_cnt = {foot_id : {} for foot_id in range(self.N_feet)}

    def set_config_gait(self, config_gait = GaitConfig) -> None:
        self.config_gait = config_gait

    def next_contact_location(self,
                              i_foot : int,
                              q : np.ndarray,
                              com_xyz : np.ndarray,
                              time : float,
                              time_cnt : float,
                              v_des : np.ndarray = np.zeros(3),
                              w_yaw : float = 0.,
                              ):
        time_cnt = round(time_cnt, 2)
        com_xy, com_z = com_xyz[:2], com_xyz[-1] - self.height_offset
        vtrack = v_des[:2]

        # R base in world frame
        w_T_b = pin.XYZQUATToSE3(q[:7])
        
        # Set roll and pitch to 0.
        rpy_vector = pin.rpy.matrixToRpy(w_T_b.rotation)
        rpy_vector[:2] = 0.0  # Set roll and pitch to 0
        R_horizontal = pin.rpy.rpyToMatrix(rpy_vector)

        # Check if location already computed
        cnt_pos_w = None
        if self.cache_cnt:
            cnt_pos_w = self.planed_cnt[i_foot].get(time_cnt, None)

        # Compute the contact location with Raibert heuristic
        # Store the planned contact
        if cnt_pos_w is None:
            cnt_pos_w = np.zeros(3)
            time_to_cnt = round(time_cnt - time, 3)
            if time_to_cnt > 0:
                cnt_pos_w[-1] = self.height_offset + self.foot_size
                hip_loc = com_xy + (R_horizontal @ self.offset_hip_b[i_foot])[0:2] + time_to_cnt * vtrack
                raibert_step = (0.5 * vtrack * self.config_gait.nominal_period * self.config_gait.stance_ratio[i_foot])
                ang_step = np.cross(0.5 * np.sqrt(com_z / RaiberContactPlanner.GRAVITY) * vtrack, [0.0, 0.0, w_yaw])
            
                cnt_pos_w[:2] = raibert_step + hip_loc + ang_step[0:2]

                # Add locations to the planned contact
                self.planed_cnt[i_foot][time_cnt] = cnt_pos_w
            
        return cnt_pos_w
    
    def remove_cnt_before(self, time : float):
        """
        Remove planned contact locations before current time.
        """
        self.planed_cnt = {
            i_foot : {t : cnt_w for t, cnt_w in d.items() if t >= time}
            for i_foot, d
            in self.planed_cnt.items()
        }