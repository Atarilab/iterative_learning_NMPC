import os
import numpy as np

from mj_pin.abstract import DataRecorder, VisualCallback
from mpc_controller.mpc import LocomotionMPC

class SteppingStonesDataRecorder_Visual(DataRecorder, VisualCallback):
    KEYS = ["q", "v", "feet_pos_w", "contact_plan_w", "gait_cycle"]
    FILE_NAME = "data.npz"
    def __init__(self, mpc : LocomotionMPC, record_dir = "", record_step = 1):
        DataRecorder.__init__(self, record_dir, record_step)
        VisualCallback.__init__(self)
        self.mpc = mpc
        self.i_gait_cycle = 0
        self.sphere_radius = 0.01
        self.n_cnt_plan = len(self.mpc.contact_planner._contact_locations)
        self.last_opt_node = -np.inf
        
    def _new_gait_cycle(self) -> bool:
        """
        Returns true when sim step corresponds to a new gait
        cycle of the NMPC.
        """
        if (self.mpc.current_opt_node >= self.i_gait_cycle * self.mpc.contact_planner.nodes_per_cycle and
            self.last_opt_node < self.mpc.current_opt_node):
            # Avoid repetitions between record and add_visuals
            self.last_opt_node = self.mpc.current_opt_node
            # Next gait cycle
            self.i_gait_cycle += 1
            return True
        return False

    def record(self, mj_data, **kwargs):
        # Record every new gait cycle
        if self._new_gait_cycle():
            # Save full contact plan
            # Save the gait cycle appart to get the target at each step
            if self.i_gait_cycle == 1:
                self.data["contact_plan_w"].append(self.mpc.contact_planner._contact_locations)
            
            # Record state position
            q, v = mj_data.qpos.copy(), mj_data.qvel.copy()
            self.data["q"].append(q)
            # Record state velocity
            self.data["v"].append(v)
            # Record feet position in world
            q_pin, v_pin = self.mpc.solver.dyn.convert_from_mujoco(q, v)
            self.mpc.solver.dyn.update_pin(q_pin, v_pin)
            feet_pos_w = self.mpc.solver.dyn.get_feet_position_w()
            self.data["feet_pos_w"].append(feet_pos_w)
            # Record all future cnt locations in the plan, keep the last one
            i = min(self.i_gait_cycle, self.n_cnt_plan) - 1
            self.data["gait_cycle"].append(i)

    def add_visuals(self, mj_data):
        if not self.record_dir: self._new_gait_cycle()
        
        i = min(self.i_gait_cycle, self.n_cnt_plan) - 1
        cnt_plan_full = self.mpc.contact_planner._contact_locations[i:]
        for pos_feet in cnt_plan_full:
            for i, pos in enumerate(pos_feet):
                self.add_sphere(pos, self.sphere_radius, self.colors.id(i))

    def reset(self) -> None:
        """
        Reset the recorder data.
        """
        self.data = {}
        for k in SteppingStonesDataRecorder_Visual.KEYS: self.data[k] = []

    def save(self) -> None:
        """
        Save the recorded data to a file in the specified directory.
        """
        if not self.record_dir:
            self.record_dir = os.getcwd()
        # Uncomment to save data
        os.makedirs(self.record_dir, exist_ok=True)
        file_path = os.path.join(self.record_dir, SteppingStonesDataRecorder_Visual.FILE_NAME)
        if os.path.exists(file_path):
            file_path = os.path.join(self.record_dir, f"data_{self.get_date_time_str()}.npz")
        
        try:
            # Uncomment to save data
            np.savez(file_path, **self.data)
            # print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")


