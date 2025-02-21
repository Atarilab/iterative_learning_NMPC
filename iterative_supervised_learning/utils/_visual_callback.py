from typing import List
import os
import numpy as np
import argparse
import mujoco

from mj_pin.abstract import MjController, VisualCallback
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description, mj_frame_pos

class MjPDController(MjController):
    def __init__(self, xml_path : str, Kp = 44., Kd = 3.):
        super().__init__(xml_path)
        self.Kp, self.Kd = Kp, Kd
        self.q_ref = None

    def get_torques(self, sim_step, mj_data):
        # Get pos, vel state
        q, v = self.get_state(mj_data)
        # Set reference as the first state
        if self.q_ref is None: self.q_ref = q[-self.nu:].copy()
        # Update torques_dof
        self.torques_dof[-self.nu:] = self.Kp * (self.q_ref - q[-self.nu:]) - self.Kd * v[-self.nu:]
        # torque map {joint name : torque value}
        torque_map = self.get_torque_map()
        
        return torque_map
    
class FeetVisualCallback(VisualCallback):
    def __init__(self, xml_path, feet_names : List[str], update_step = 1):
        super().__init__(update_step)
        self.xml_path = xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.feet_names = feet_names
        self.radius = 0.03

    def add_visuals(self, mj_data):
        for i, f_name in enumerate(self.feet_names):
            pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            # Use parent class to add specific geometries.
            # Use self.colors to easily handle colors 
            if i < 2:
                self.add_sphere(pos, self.radius, self.colors.id(i))
            else:
                rot_euler = np.zeros(3)
                self.add_box(pos, rot_euler, [self.radius] * 3, self.colors.BLUE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a robot with optional recording and visualization.")
    parser.add_argument("--robot_name", type=str, default="go2", help="Name of the robot to simulate.")
    args = parser.parse_args()

    # Load robot information and paths
    robot_description = get_robot_description(args.robot_name)
    mj_eeff_frame_name = ["FL", "FR", "RL", "RR"]

    # Load the simulator
    # Will start two threads: one for the viewer and one for the physics simulation.
    # Viewer and physics are updated at different rate.
    sim = Simulator(robot_description.xml_scene_path, sim_dt=1e-3, viewer_dt=1/40)

    # PD Controller, called every simulation step
    pd_controller = MjPDController(robot_description.xml_path)

    # Visual callback on the viewer, called every 2 viewer step
    visual_update_freq = 2
    vis_feet_pos = FeetVisualCallback(
        robot_description.xml_path,
        mj_eeff_frame_name,
        visual_update_freq
        )

    # Run the simulation with the provided controller etc.
    sim.run(controller=pd_controller,
            visual_callback=vis_feet_pos)