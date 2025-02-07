import os
from dataclasses import dataclass
import mujoco
import numpy as np

from mj_pin.abstract import DataRecorder, VisualCallback, Controller
from mj_pin.simulator import Simulator
from mpc_controller.mpc import LocomotionMPC
from .scene.stepping_stones import MjSteppingStones
from .graph_stepping_stones import SteppingStonesGraph
from .utils.a_star import a_star_search, reconstruct_path
from .utils.config import ConfigBase

### DataRecorder
class SteppingStonesDataRecorder_Visual(DataRecorder, VisualCallback):
    KEYS = ["time", "q", "v", "feet_pos_w", "target_cnt_w"]
    def __init__(self, mpc : LocomotionMPC, record_dir = "", record_step = 1):
        DataRecorder.__init__(self, record_dir, record_step)
        VisualCallback.__init__(self)
        self.mpc = mpc
        self.i_gait_cycle = 0
        self.sphere_radius = 0.01
        self.n_cnt_plan = len(self.mpc.contact_planner._contact_locations)
        
    def _new_gait_cycle(self) -> bool:
        """
        Returns true when sim step corresponds to a new gait
        cycle of the NMPC.
        """
        return self.mpc.current_opt_node >= self.i_gait_cycle * self.mpc.contact_planner.nodes_per_cycle
        
    def record(self, mj_data, **kwargs):
        # Record every new gait cycle
        if self._new_gait_cycle():           
            # Record state position
            self.data["q"].append(mj_data.qpos.copy())
            # Record state velocity
            self.data["v"].append(mj_data.qvel.copy())
            # Record feet position in world
            feet_pos_w = self.mpc.solver.dyn.get_feet_position_w()
            self.data["feet_pos_w"].append(feet_pos_w)
            # Record all future cnt locations in the plan, keep the last one
            i = min(self.i_gait_cycle, self.n_cnt_plan - 1)
            cnt_plan_full = self.mpc.contact_planner._contact_locations[i:]
            self.data["target_cnt_w"].append(cnt_plan_full)
            
            # Next gait cycle
            self.i_gait_cycle += 1
             
    def add_visuals(self, mj_data):
        i = min(self.i_gait_cycle, self.n_cnt_plan - 1)
        cnt_plan_full = self.mpc.contact_planner._contact_locations[i:]
        for pos_feet in cnt_plan_full:
            for i, pos in enumerate(pos_feet):
                self.add_sphere(pos, self.sphere_radius, self.colors.id(i))

    def reset(self) -> None:
        """
        Reset the recorder data.
        """
        self.data = dict.fromkeys(SteppingStonesDataRecorder_Visual.KEYS, list())

    def save(self) -> None:
        """
        Save the recorded data to a file in the specified directory.
        """
        if not self.record_dir:
            self.record_dir = os.getcwd()
        # Uncomment to save data
        # os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            print(self.data["time"][:10])
            # Uncomment to save data
            # np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

@dataclass  
class SearchConfig(ConfigBase):
    xml_path : str = "",
    feet_frames_mj : list[str] = [],
    grid_size : tuple[int, int] = (10, 10),
    spacing : tuple[float, float] = (0.19, 0.19),
    size_ratio : tuple[float, float] = (0.45, 0.45),
    randomize_pos_ratio : float = 0.75,
    randomize_height_ratio : float = 0.1,
    n_remove : int = 50,
    height : float = 0.2,
    shape : str = "box",
    max_step_size : float = 0.31,
    max_foot_displacement : float = 0.265,
    
    def __post_init__(self):
        self.file_name = "search_config.yaml"


def search_contact_plan(search_config : SearchConfig) -> tuple[MjSteppingStones, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    
    cfg = search_config
    
    ### Init stepping stones   
    stones = MjSteppingStones(
        grid_size=cfg.grid_size,
        spacing=cfg.spacing,
        height=cfg.height,
        size_ratio=cfg.size_ratio,
        randomize_pos_ratio=cfg.randomize_pos_ratio,
        randomize_height_ratio=cfg.randomize_height_ratio,
        shape=cfg.shape
        )
    
    ### Search until a solution is found
    mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    mj_data = mujoco.MjData(mj_model)
    path = []
    while not path:
        # Setup start and goal
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)
        start, goal, q0, v0 = stones.setup_start_goal(mj_model, mj_data, cfg.feet_frames_mj, cfg.n_remove, randomize_state=True, random_goal=True)
        
        # Construct graph
        n_feet = len(cfg.feet_frames_mj)
        graph = SteppingStonesGraph(
            stones.positions,
            n_feet=n_feet,
            max_step_size=cfg.max_step_size,
            max_foot_displacement=cfg.max_foot_displacement
        )
        
        # Define heuristic
        def distance_between_centers(positions, node_from, node_to):
            return np.linalg.norm(np.mean(positions[[node_from]] - positions[[node_to]], axis=0))
        heuristic = lambda node_from, node_to : distance_between_centers(stones.positions, node_from, node_to)

        # A*
        start_node, goal_node = tuple(start), tuple(goal)
        came_from, _ = a_star_search(
            graph,
            start_node,
            goal_node,
            heuristic,
            max_time=0.5
        )
        path = reconstruct_path(came_from, start_node, goal_node)
        
    return stones, path, q0, v0

@dataclass
class RunConfig(ConfigBase):
    xml_path : str = "",
    n_run : int = 1,
    n_cores : int = 1,
    sim_dt : float = 1.0e-3,
    
def run_contact_plan(
    controller : Controller,
    sim_time : float,
    stones : MjSteppingStones,
    path : list[tuple[int, ...]],
    q0 : np.ndarray,
    v0 : np.ndarray,
    run_config : RunConfig,
    use_viewer : bool = False,
    record_video : bool = False,
    ) -> bool:
    
    sim = Simulator(run_config.xml_path, run_config.sim_dt, verbose=False)
    start, goal = path[0], path[-1]
    stones.setup_sim(sim, q0, v0, start, goal)
    
    # Run sim
    sim.run(sim_time,
            use_viewer=use_viewer,
            controller=controller,
            record_video=record_video,
            )
    
    return not(sim.collided)
    