import numpy as np
import math
from bisect import bisect_right
from typing import List, Tuple
from ..config.config_abstract import GaitConfig
from .contact_plan import CyclicContactPlan

class GaitPlanner():
    def __init__(self,
                 feet_frame_names : List[str],
                 dt_nodes : float,
                 config_gait : GaitConfig,
                 ) -> None:
        
        self.config = config_gait
        self.feet_frame_names = feet_frame_names
        self.nodes_per_cycle = round(self.config.nominal_period / dt_nodes)
        
        self.contact_plan = CyclicContactPlan(feet_frame_names)
        self.in_cnt_intervals = {foot : [] for foot in self.feet_frame_names}
        self.swing_intervals = {foot : [] for foot in self.feet_frame_names}
        self.init_gait_cycle()

    def is_in_cnt_phase(self, foot : str, phase : float) -> bool:
        for cnt_interval in self.in_cnt_intervals.get(foot, []):
            if cnt_interval[0] <= phase < cnt_interval[1]:
                return True
        return False

    def is_in_cnt(self, foot : str, i_node : int) -> bool:
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle
        phase = round(i_node_cycle / self.nodes_per_cycle, 3)
        return self.is_in_cnt_phase(foot, phase)
    
    def init_gait_cycle(self):
        phase_offset_feet = self.config.phase_offset
        stance_ratio_feet = self.config.stance_ratio

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
        for foot, mk_cnt, bk_cnt in zip(self.feet_frame_names,
                                        make_cnt_phase,
                                        break_cnt_phase):
            if mk_cnt < bk_cnt:
                # Contact
                interval_cnt = (mk_cnt, bk_cnt)
                self.in_cnt_intervals[foot].append(interval_cnt)

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

                if interval_cnt1[0] != interval_cnt1[1]:
                    self.in_cnt_intervals[foot].append(interval_cnt1)
                if interval_cnt2[0] != interval_cnt2[1]:
                    self.in_cnt_intervals[foot].append(interval_cnt2)

                # Swing
                interval_swing = (bk_cnt, mk_cnt)
                self.swing_intervals[foot].append(interval_swing)

    def init_contact_plan(self):

        for switch in self.switch_phase:
            # Find the feet in contact for each switch
            feet_in_cnt = [foot
                           for foot in self.feet_frame_names
                           if self.is_in_cnt_phase(foot, switch)]
            cnt_pos = [np.zeros(3)] * len(feet_in_cnt)

            # Add to the plan
            self.contact_plan.add(cnt_pos, feet_in_cnt)

    def get_swing_start_peak_end(self, foot: str, i_node: int) -> Tuple[int, int, int]:
        """
        Return start, peak, and end optimization nodes for
        the swing phase of a given foot in the current gait cycle.
        Peak is defined as the midpoint between start and end. 
        """
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle

        # Find the current and next swing interval
        for (start, end) in self.swing_intervals.get(foot, []):
            start_node = math.floor(start * self.nodes_per_cycle)
            end_node = math.ceil(end * self.nodes_per_cycle)

            # Check if the current phase falls within the current swing interval
            if start_node <= i_node_cycle < end_node:
                # Translate to the current optimization window
                peak_node = (start_node + end_node) // 2 - i_node_cycle
                start_node = max(0, start_node - i_node_cycle)
                end_node = max(0, end_node - i_node_cycle)

                return start_node, peak_node, end_node

        # Default return if in contact
        return -1, -1, -1
    
    def next_switch_in(self, i_node : int) -> int:
        """
        Get how many nodes before next switch in the gait.
        <i_node> is the current optimization node.
        """
        # Calculate current phase in the cycle
        i_node_cycle = i_node % self.nodes_per_cycle

        # Get next switch
        switch_node = [math.floor(phase * self.nodes_per_cycle) for phase in self.switch_phase]
        i_next_switch = bisect_right(switch_node, i_node_cycle)
        i_next_switch = i_next_switch % len(self.switch_phase)

        next_switch_node = switch_node[i_next_switch]
        next_switch_in = abs(next_switch_node - i_node_cycle)

        return next_switch_in

if __name__ == "__main__":
    from config.quadruped.mpc_gait import QuadrupedTrot, QuadrupedJump
    feet_names = ["FL", "FR", "RL", "RR"]
    dt_nodes = 1e-2
    gait_cfg = QuadrupedTrot()
    gait_planer = GaitPlanner(feet_names, dt_nodes, gait_cfg)