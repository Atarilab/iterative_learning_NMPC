import mujoco
import os
import numpy as np

from .config import SearchConfig, RunConfig
from .data_recorder import SteppingStonesDataRecorder_Visual
from mj_pin.simulator import Simulator
from mj_pin.abstract import Controller
from search.scene.stepping_stones import MjSteppingStones
from search.graph_stepping_stones import SteppingStonesGraph
from search.utils.a_star import a_star_search, reconstruct_path

def search_contact_plan(search_config : SearchConfig, record_dir : str = "") -> tuple[MjSteppingStones, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    
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
        
    if record_dir:
        search_config.save(record_dir)
        stones.save(record_dir)
        # Save initial conditions
        file_name = os.path.join(record_dir, "initial_state.npz")
        np.savez(file_name,
                 q0=q0,
                 v0=v0,
                 )
        # Save path
        file_name = os.path.join(record_dir, "path_found.npz")
        np.savez(file_name,
                 path=np.array(path, dtype=np.int32),
                 )
        
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
    record_dir : str = "",
    ) -> bool:
    
    sim = Simulator(run_config.xml_path, run_config.sim_dt, verbose=False)
    start, goal = path[0], path[-1]
    stones.setup_sim(sim, q0, v0, start, goal)
    
    if run_config.collision:
        allowed_collision = run_config.feet_frames_mj + sim.edit.name_allowed_collisions
    else:
        allowed_collision = []
    
    data_recorder = None
    if record_dir or use_viewer:
        data_recorder = SteppingStonesDataRecorder_Visual(controller, record_dir)
    
    # Run sim
    sim.run(sim_time,
            use_viewer=use_viewer,
            controller=controller,
            record_video=record_video,
            data_recorder=data_recorder if record_dir else None,
            visual_callback=data_recorder,
            allowed_collision=allowed_collision
            )
    
    # Delete data recorded
    if sim.collided:
        file_name = os.path.join(record_dir, SteppingStonesDataRecorder_Visual.FILE_NAME)
        if os.path.exists(file_name):
            os.remove(file_name)
        return False
    
    return True