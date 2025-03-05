from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from math import ceil
import pinocchio as pin

from ..config.config_abstract import GaitConfig

class GaitPlanner(ABC):
    def __init__(self,
                 feet_frame_names : List[str],
                 dt_nodes : float,
                 config_gait : GaitConfig,
                 ):
        super().__init__()
        self.feet_frame_names = feet_frame_names
        self.n_foot = len(self.feet_frame_names)
        self.dt_nodes = dt_nodes
        self.config_gait = config_gait

        # Phase interval with foot in contact
        self.cnt_intervals = {foot : [] for foot in self.feet_frame_names}
        # Phase interval with foot swinging
        self.swing_intervals = {foot : [] for foot in self.feet_frame_names}
        # Gait sequence
        self.nodes_per_cycle = round(self.config_gait.nominal_period / dt_nodes)
        self.gait_sequence = np.zeros((self.n_foot, self.nodes_per_cycle), dtype=np.int8)
        self.peak_swing = np.zeros((self.n_foot, self.nodes_per_cycle), dtype=np.int8)
        self.switch_cnt = np.zeros((self.n_foot, self.nodes_per_cycle), dtype=np.int8)
        self._init_gait_cycle()

    def _is_in_cnt_phase(self, foot : str, phase : float) -> bool:
        for cnt_interval in self.cnt_intervals.get(foot, []):
            if cnt_interval[0] <= phase < cnt_interval[1]:
                return True
        return False

    def _is_in_cnt(self, foot : str, i_node : int) -> bool:
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle
        phase = round(i_node_cycle / self.nodes_per_cycle, 3)
        return self._is_in_cnt_phase(foot, phase)
    
    def _init_gait_cycle(self):
        phase_offset_feet = self.config_gait.phase_offset
        stance_ratio_feet = self.config_gait.stance_ratio

        # Foot make contact (phase)
        make_cnt_phase = phase_offset_feet
        # Foot break contact (phase)
        break_cnt_phase = ((phase_offset_feet + stance_ratio_feet) % 1.).round(2)

        # Switch phase
        self.switch_phase = np.unique(
            np.concatenate((break_cnt_phase, make_cnt_phase)),
            return_inverse = False
        )

        # Init contact and swing intervals
        # Phase interval during which the foot is in contact or swinging
        for foot_id, (foot, mk_cnt, bk_cnt) in enumerate(zip(self.feet_frame_names,
                                                        make_cnt_phase,
                                                        break_cnt_phase)):
            if mk_cnt < bk_cnt:
                # Contact
                interval_cnt = (mk_cnt, bk_cnt)
                self.cnt_intervals[foot].append(interval_cnt)

                # Fill gait sequence
                start_idx = ceil(mk_cnt * self.nodes_per_cycle)
                end_idx = ceil(bk_cnt * self.nodes_per_cycle)
                self.gait_sequence[foot_id, start_idx:end_idx] = 1

                # Swing
                interval_swing1 = (bk_cnt, 1.)
                interval_swing2 = (0., mk_cnt)
                
                if interval_swing1[0] != interval_swing1[1]:
                    self.swing_intervals[foot].append(interval_swing1)
                if interval_swing2[0] != interval_swing2[1]:
                    self.swing_intervals[foot].append(interval_swing2)

            else:
                # Contact
                interval_cnt1 = (mk_cnt, 1.)
                interval_cnt2 = (0., bk_cnt)

                start_idx = ceil(mk_cnt * self.nodes_per_cycle)
                end_idx = ceil(bk_cnt * self.nodes_per_cycle)
                self.gait_sequence[foot_id, start_idx:] = 1
                self.gait_sequence[foot_id, :end_idx] = 1

                if interval_cnt1[0] != interval_cnt1[1]:
                    self.cnt_intervals[foot].append(interval_cnt1)
                if interval_cnt2[0] != interval_cnt2[1]:
                    self.cnt_intervals[foot].append(interval_cnt2)

                # Swing
                interval_swing = (bk_cnt, mk_cnt)
                self.swing_intervals[foot].append(interval_swing)

            self.switch_cnt[foot_id, start_idx] = 1
            self.switch_cnt[foot_id, end_idx] = -1

            # Peak in the middle
            peak = end_idx + (start_idx+end_idx) // 2
            o = abs(peak - end_idx) // 3
            self.peak_swing[foot_id, end_idx+o:start_idx-o] = 1
            if peak < len(self.peak_swing):
                self.peak_swing[foot_id, peak] = 0.75

    def get_contacts(self, i_node : int, n_nodes) -> np.ndarray:
        """
        Get gait sequence for the next <n_nodes> starting at node <i_node>.

        Returns:
            - array of shape [n_foot, n_nodes]. 1 is contact, 0 is swing.
        """
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle
        
        n_repeat_cyle = n_nodes // self.nodes_per_cycle + 2
        gait_cycle_extended = np.tile(self.gait_sequence, (1, n_repeat_cyle))

        return gait_cycle_extended[:, i_node_cycle:i_node_cycle+n_nodes]
    
    def get_peaks(self, i_node : int, n_nodes) -> np.ndarray:
        """
        Get peak (middle point of each swing phase) for the next <n_nodes> starting at node <i_node>.

        Returns:
            - array of shape [n_foot, n_nodes]. 1 is peak.
        """
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle
        
        n_repeat_cyle = n_nodes // self.nodes_per_cycle + 2
        peak_cycle_extended = np.tile(self.peak_swing, (1, n_repeat_cyle))

        return peak_cycle_extended[:, i_node_cycle:i_node_cycle+n_nodes]

    def get_make_break_contacts(self, i_node : int, n_nodes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get make/break contact indices for the next <n_nodes> starting at node <i_node>.

        Returns:
            - arrays of shape [n_foot, n_nodes]. 1 is making/break contact, 0 otherwise.
        """
        # Calculate current phase in the cycle
        i_cycle = i_node % self.nodes_per_cycle
        
        mk = np.where(self.switch_cnt == 1, self.switch_cnt, 0)
        bk = np.where(self.switch_cnt == -1, -self.switch_cnt, 0)
        
        n_repeat_cyle = n_nodes // self.nodes_per_cycle + 2
        mk_extended = np.tile(mk, (1, n_repeat_cyle))
        bk_extended = np.tile(bk, (1, n_repeat_cyle))

        return mk_extended[:, i_cycle:i_cycle+n_nodes], bk_extended[:, i_cycle:i_cycle+n_nodes]

class ContactPlanner(GaitPlanner):
    def __init__(
        self,
        feet_frame_names : List[str],
        dt_nodes : int,
        config_gait : GaitConfig,
        ):
        super().__init__(feet_frame_names, dt_nodes, config_gait)
    
    def get_locations(self, i_node : int, n_nodes : int) -> np.ndarray:
        pass

class RaiberContactPlanner(ContactPlanner):
    GRAVITY = 9.81
    V_TRACKING = 0.05
    def __init__(
        self,
        feet_frame_names : List[str],
        dt_nodes : int,
        config_gait : GaitConfig,
        offset_hip_b : np.ndarray,
        x_offset : float = 0.,
        y_offset : float = 0.,
        foot_size : float = 0.,
        height_offset : float = 0.,
        cache_cnt : bool = True,
        ):
        super().__init__(feet_frame_names, dt_nodes, config_gait)
        """
        Contact planner (locations) based on Raibert heuristic.

        Args:
            feet_frame_names : List[str]: feet frames name.
            dt_nodes : int: dt between two optimization nodes.
            config_gait : GaitConfig: gait config.
            offset_hip_b (np.ndarray): Initial offsets of the hips relative to the base 
                in the base frame, shape (N_feet, 3).
            config_gait (GaitConfig): Gait configuration specifying gait parameters such as 
                stance ratio, nominal period, and phase offsets.
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
        self.foot_size = foot_size
        self.cache_cnt = cache_cnt

        self.offset_hip_b = offset_hip_b
        if self.n_foot == 4:
            self.offset_hip_b[:, 0] += np.array([x_offset, x_offset, -x_offset, -x_offset])
            self.offset_hip_b[:, 1] += np.array([y_offset, -y_offset, y_offset, -y_offset])

        self.default_cnt_w = np.zeros(3)
        self.height_offset = height_offset
        # {foot : {i_node : pos}}
        self.planed_cnt = {foot_id : {} for foot_id in range(self.n_foot)}

        self.pos = None
        self.v_w = None
        self.euler_rpy = None
        self.com_xyz = None
        self.v_des = None
        self.w_yaw = None

    def set_state(self,
                  pos : np.ndarray,
                  v_w : np.ndarray,
                  euler_rpy : np.ndarray,
                  com_xyz : np.ndarray,
                  v_des : np.ndarray = np.zeros(3),
                  w_yaw : float = 0.,
                  ):
        self.pos = pos
        self.v_w = v_w
        self.euler_rpy = euler_rpy
        self.com_xyz = np.array(com_xyz)
        self.v_des = np.array(v_des)
        self.w_yaw = w_yaw

    def remove_cnt_before(self, i_node : int):
        """
        Remove planned contact locations before current node.
        """
        self.planed_cnt = {
            i_foot : {node : cnt_w for node, cnt_w in d.items() if node >= i_node}
            for i_foot, d
            in self.planed_cnt.items()
        }

    def get_locations(self, i_node: int, n_nodes: int) -> np.ndarray:
        """
        Compute the contact locations for the next n_nodes optimization steps.

        Args:
            i_node (int): Starting node.
            n_nodes (int): Number of nodes to plan for.

        Returns:
            np.ndarray: Contact locations of shape (N_feet, n_nodes, 3).
        """
        contact_locations = np.zeros((self.n_foot, n_nodes, 3))  # Default 0. for invalid locations
        mk_cnt, _ = self.get_make_break_contacts(i_node, n_nodes)
        mk_cnt_id = np.argwhere(mk_cnt == 1)

        # Process state
        com_xy, com_z = self.com_xyz[:2], self.com_xyz[-1] - self.height_offset
        vtrack = self.v_des[:2]
        vtrack_direction = vtrack / np.linalg.norm(vtrack)

        # Set roll and pitch to 0.
        rpy_des = np.array([0., 0., self.euler_rpy[2]])
        R_horizontal = pin.rpy.rpyToMatrix(rpy_des)

        for (i_foot, i_mk) in mk_cnt_id:
            # Absolute node of the contact
            abs_i_node = i_node + i_mk

            # Check if already cached
            if self.cache_cnt:
                cnt_pos_w = self.planed_cnt[i_foot].get(abs_i_node, None)
                if cnt_pos_w is not None:
                    contact_locations[i_foot, i_mk:] = cnt_pos_w
                    continue
            
            # Compute next contact locations
            cnt_pos_w = np.zeros(3)
            time_to_cnt = round(i_mk * self.dt_nodes, 3)

            if time_to_cnt >= 0:
                t_stance = self.config_gait.nominal_period * self.config_gait.stance_ratio[i_foot]
                # Compute the Raibert heuristic
                hip_loc = com_xy + (R_horizontal @ self.offset_hip_b[i_foot])[0:2] + vtrack * time_to_cnt * (1 + self.config_gait.stance_ratio[i_foot])
                step_adjustment = 0.1 * (vtrack - self.v_w[:2])
                raibert_step = (0.5 * vtrack * t_stance)
                angular_adjustment = np.cross(
                    0.5 * np.sqrt(com_z / RaiberContactPlanner.GRAVITY) * vtrack,
                    [0.0, 0.0, self.w_yaw]
                )
                cnt_pos_w[:2] = hip_loc + step_adjustment + raibert_step + angular_adjustment[:2]
                cnt_pos_w[-1] = self.foot_size

                # Add locations to the planned contact
                contact_locations[i_foot, i_mk:] = cnt_pos_w[None, :]

                if self.cache_cnt:
                    self.planed_cnt[i_foot][abs_i_node] = cnt_pos_w
        return contact_locations
    
class CustomContactPlanner(ContactPlanner):
    def __init__(self, feet_frame_names, dt_nodes, config_gait):
        super().__init__(feet_frame_names, dt_nodes, config_gait)
        self._contact_locations = None
        self.contact_locations_full = None
        self._contact_sequence = None
        self.contact_sequence_full = None
        self.n_full = 0
        
    def set_contact_locations(self, contact_locations : np.ndarray) -> None:
        """
        Set contact locations to reach at the end of each gait cycle.
        
        Args:
            contact_locations (np.ndarray): shape [N, feet, 3]
        """
        if not (contact_locations.shape[-1] == 3 and
                contact_locations.shape[-2] == self.n_foot and 
                len(contact_locations.shape) == 3):
            raise ValueError(f"contact_locations: incorrect shape ({print(contact_locations.shape)}).")

        # Save original contact plan
        self._contact_locations = contact_locations
        
        # Repeat last location
        N_REPEAT = 3
        last_locations = np.repeat(contact_locations[-1, None], N_REPEAT, axis=0)
        contact_locations_extended = np.concatenate((contact_locations, last_locations), axis=0)
        # Contact plan at dt interval
        self.contact_locations_full = np.repeat(contact_locations_extended, self.nodes_per_cycle, axis=0).transpose(1, 0, 2)
        self.n_full = self.contact_locations_full.shape[1]

    def get_locations(self, i_node, n_nodes):
        last_node = i_node + n_nodes
        # Output shape [n_feet, n_nodes, 3]
        if last_node < self.n_full:
            return self.contact_locations_full[:, i_node:last_node, :].copy()
        # Take only the last contact locations
        else:
            return self.contact_locations_full[:, -n_nodes:, :].copy()
        
    def set_periodic_sequence(self, cnt_sequence : np.ndarray) -> None:
        """
        Set contact sequence for the next optimization problem.
        
        Args:
            cnt_sequence (np.ndarray): shape [feet, N]
        """
        if cnt_sequence.shape != self.gait_sequence.shape:
            raise ValueError(f"Invalid cnt_sequence shape, should be of shape {self.gait_sequence.shape}.")
        
        self.gait_sequence = cnt_sequence.copy()
        self.contact_sequence_full = None
    
    def get_contacts(self, i_node, n_nodes) -> np.ndarray:
        if self.contact_sequence_full is not None and n_nodes + i_node <= len(self.contact_locations_full):
            return self.contact_sequence_full[:, i_node:i_node+n_nodes]
        else:
            return super().get_contacts(i_node, n_nodes)
