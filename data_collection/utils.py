import mujoco
import numpy as np

from .config import SearchConfig, RunConfig
from mj_pin.simulator import Simulator
from mj_pin.abstract import Controller
from search.scene.stepping_stones import MjSteppingStones
from search.graph_stepping_stones import SteppingStonesGraph
from search.utils.a_star import a_star_search, reconstruct_path

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
    