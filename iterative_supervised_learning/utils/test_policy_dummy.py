import os
from typing import List, Dict
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder, Controller # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import get_robot_description, mj_joint_name2act_id
# from mpc_controller.mpc import LocomotionMPC
import mujoco

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.

# VisualCallback
class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step = 1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def add_visuals(self, mj_data):
        # Contact locations
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.: continue
                self.add_sphere(pos, self.radius, self.colors.id(i))

        # Base reference
        BLACK = self.colors.name("black")
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)
        
        # Base terminal reference
        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)
        

class DummyController(Controller):
    def __init__(self, control_file: str, joint_name2act_id: Dict[str, int]) -> None:
        super().__init__()
        self.control_file = control_file
        self.controls = None
        self.current_step = -1  # Start at -1 to apply the first control correctly
        self.joint_name2act_id = joint_name2act_id
        self.load_controls()

    def load_controls(self) -> None:
        if not os.path.exists(self.control_file):
            raise FileNotFoundError(f"Control file not found: {self.control_file}")
        data = np.load(self.control_file)
        if "ctrl" not in data:
            raise ValueError(f"Control file does not contain 'ctrl' data: {self.control_file}")
        self.controls = np.array(data["ctrl"], dtype=np.float64)  # Enforce float64 precision
        print("Loaded controls with precision:", self.controls.dtype)

    def get_torques(self, step: int, mj_data) -> Dict[str, float]:
        if self.controls is None or self.current_step >= len(self.controls) - 1:
            return {}
        self.current_step += 1
        torque = self.controls[self.current_step]
        print(f"Step {step}: Applied control torques (high precision): {torque}")
        # input()
        return self.create_torque_map(torque)

    def create_torque_map(self, torques: np.ndarray) -> Dict[str, float]:
        torque_map = {
            j_name: float(torques[joint_id])  # Ensure float64 precision is maintained
            for j_name, joint_id in self.joint_name2act_id.items()
            if joint_id < len(torques)
        }
        print(torque_map)
        return torque_map


class StateDataRecorder(DataRecorder):
    def __init__(self, record_dir: str = "./data", record_step: int = 1) -> None:
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        self.data = {"time": [], "q": [], "v": [], "ctrl": []}

    def save(self) -> None:
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)

        file_path = os.path.join(self.record_dir, "simulation_data.npz")
        np.savez(file_path, **self.data)
        print(f"Data successfully saved to {file_path}")

    def record(self, mj_data) -> None:
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(mj_data.qpos.copy())
        self.data["v"].append(mj_data.qvel.copy())
        self.data["ctrl"].append(mj_data.ctrl.copy())


def run_simulation_no_controller(
    robot_name: str = "go2",
    sim_time: float = 5.0,
    record_video: bool = False,
    save_data: bool = True,
    record_dir: str = "./data/",
    visualize: bool = True
):
    robot_desc = get_robot_description(robot_name)
    # visual_callback = ReferenceVisualCallback() if visualize else None
    # data_recorder = StateDataRecorder(record_dir) if save_data else None

    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.set_high_quality()
    sim.run(
        sim_time=sim_time,
        use_viewer=True,
        controller=None,
        record_video=record_video
    )

def run_simulation_with_dummy_controller(
    robot_name: str = "go2",
    control_file: str = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_26_2025_10_35_39/dataset/experiment/simulation_data_02_26_2025_10_35_57.npz",
    sim_time: float = 5.0,
    record_video: bool = False,
    save_data: bool = True,
    record_dir: str = "./data/",
    visualize: bool = True
):
    robot_desc = get_robot_description(robot_name)
    # visual_callback = ReferenceVisualCallback() if visualize else None
    # data_recorder = StateDataRecorder(record_dir) if save_data else None

    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.set_high_quality()

    #################################
    # data = np.load(control_file)
    # print("Control data type:", data["ctrl"].dtype)  # Should ideally be float64
    # input()
    
    # Initialize the simulator to populate joint_name2act_id
    # sim._init_model_data()
    controller = DummyController(control_file, sim.joint_name2act_id)
    
    # initial_state = np.load(control_file)
    # if "q" in initial_state and "v" in initial_state:
    #     sim.set_initial_state(q0=initial_state["q"][0], v0=initial_state["v"][0])

    sim.setup()
    sim._init_model_data()
    sim.run(
        sim_time=sim_time,
        use_viewer=True,
        controller=controller,
        # visual_callback=visual_callback,
        # data_recorder=data_recorder,
        record_video=record_video
    )
    
    
    
    
# if __name__ == "__main__":
#     run_simulation_no_controller(
#         robot_name="go2",
#         sim_time=10.0,
#         record_video=False,
#         save_data=False,
#         record_dir="./data/",
#         visualize=True
#     )

if __name__ == "__main__":
    run_simulation_with_dummy_controller(
        robot_name="go2",
        sim_time=4.0,
        record_video=False,
        save_data=True,
        record_dir="./data/",
        visualize=True
    )
