import os
from typing import Dict
import numpy as np
from mj_pin.abstract import Controller
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description, mj_joint_name2act_id
import mujoco

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.


class DummyController(Controller):
    def __init__(self, control_file: str, joint_name2dof: Dict[str, int]) -> None:
        super().__init__()
        self.control_file = control_file
        self.controls = None
        self.current_step = -1  # Start at -1 to apply the first control correctly
        self.joint_name2dof = joint_name2dof
        self.load_controls()

    def load_controls(self) -> None:
        if not os.path.exists(self.control_file):
            raise FileNotFoundError(f"Control file not found: {self.control_file}")
        
        data = np.load(self.control_file)
        
        if "ctrl" not in data:
            raise ValueError(f"Control file does not contain 'ctrl' data: {self.control_file}")
        
        self.controls = np.array(data["ctrl"], dtype=np.float64)  # Ensure float64 precision
        print("Loaded controls with shape:", self.controls.shape, "and precision:", self.controls.dtype)

    def compute_torques_dof(self, mj_data) -> None:
        if self.controls is None or self.current_step >= len(self.controls) - 1:
            self.torques_dof = np.zeros(len(self.joint_name2act_id))  # Zero torques if out of bounds
            print("No more controls available, setting torques to zero.")
            return
        
        self.current_step += 1
        torque = self.controls[self.current_step]
        
        if len(torque) != len(self.joint_name2dof):
            raise ValueError(f"Control file torque dimension {len(torque)} does not match expected {len(self.joint_name2act_id)}")
        
        self.torques_dof = torque
        print(self.torques_dof)
        print(f"Step {self.current_step}: Applied control torques (high precision): {torque}")
        # input()
        
    def get_torque_map(self) -> Dict[str, float]:
        # print(self.joint_name2dof)
        torque_map = {
            j_name: self.torques_dof[dof_id]
            for j_name, dof_id in self.joint_name2dof.items()
        }
        
        # print("torque map = ", torque_map)
        # input()
        return torque_map


def run_simulation_with_dummy_controller(
    robot_name: str = "go2",
    control_file: str = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_26_2025_10_35_39/dataset/experiment/simulation_data_02_26_2025_10_35_57.npz",
    sim_time: float = 5.0,
    record_video: bool = False,
    visualize: bool = True
):
    # Initialize robot description
    robot_desc = get_robot_description(robot_name)
    
    # Initialize simulator
    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    
    # Initialize model data and joint mapping
    sim.setup()
    sim._init_model_data()  # Important to initialize the model before using joint mappings
    joint_name2act_id = mj_joint_name2act_id(sim.mj_model)
    print("Joint to Actuator ID Mapping:", joint_name2act_id)
    input()
    
    # Initialize the DummyController
    controller = DummyController(control_file, joint_name2act_id)
    
    # torque_map = controller.get_torque_map()
    # print("torque map = ", torque_map)
    # input()
    
    # Run the simulation
    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=controller,
        record_video=record_video
    )



if __name__ == "__main__":
    run_simulation_with_dummy_controller(
        robot_name="go2",
        sim_time=4.0,
        record_video=False,
        visualize=True
    )
