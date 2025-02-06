import numpy as np
import os
import time
from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mj_pin.abstract import VisualCallback, DataRecorder

from search.data_utils import SteppingStonesDataRecorder_Visual
from mpc_controller.mpc import LocomotionMPC
from scene.stepping_stones import MjSteppingStones
from search.utils.a_star import a_star_search, reconstruct_path
from search.graph_stepping_stones import SteppingStonesGraph
from main import ReferenceVisualCallback

SIM_DT = 1e-3
ROBOT_NAME = "go2"
HEIGHT = 0.2
Node = tuple[int, int, int, int]

robot_description = get_robot_description("go2")
feet_frames = ["FL", "FR", "RL", "RR"]
sim = Simulator(robot_description.xml_scene_path)
stones = MjSteppingStones(
    grid_size=(25, 25),
    spacing=(0.19, 0.19),
    height=HEIGHT,
    size_ratio=(0.45, 0.45),
    randomize_pos_ratio=0.75,
    randomize_height_ratio=0.1,
    shape="box")

sim._init_model_data()
start, goal, q0, v0 = stones.setup_start_goal(sim.mj_model, sim.mj_data, feet_frames, 0, randomize_state=False, random_goal=True)
stones.setup_sim(sim, q0, v0, start, goal)

graph = SteppingStonesGraph(
    stones.positions[stones.id_kept],
    n_feet=len(feet_frames),
    max_step_size=0.31,
    max_foot_displacement=0.265
)

def h(positions, node_from : Node, node_to : Node):
    return np.linalg.norm(np.mean(positions[[node_from]] - positions[[node_to]], axis=0))

heuristic = lambda node_from, node_to : h(stones.positions, node_from, node_to)
    
start_node, goal_node = tuple(start), tuple(goal)
t = time.time()
came_from, cost_so_far = a_star_search(
    graph,
    start_node,
    goal_node,
    heuristic,
)
print("Search time:", time.time() - t, "s")

path = reconstruct_path(came_from, start_node, goal_node)
            
# MPC Controller
robot_desc = get_robot_description(ROBOT_NAME)
feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

mpc = LocomotionMPC(
    path_urdf=robot_desc.urdf_path,
    feet_frame_names = feet_frame_names,
    robot_name=ROBOT_NAME,
    joint_ref = robot_desc.q0,
    gait_name="slow_trot",
    contact_planner="custom",
    height_offset=HEIGHT,
    interactive_goal=False,
    sim_dt=SIM_DT,
    print_info=False,
    )
mpc.contact_planner.set_contact_locations(stones.positions[path])

vis_callback = ReferenceVisualCallback(mpc)
data_recorder_visual = SteppingStonesDataRecorder_Visual(mpc, "./data/")
sim.vs.set_high_quality()
sim.vs.track_obj = "base"
sim.run(
    use_viewer=True,
    controller=mpc,
    data_recorder=data_recorder_visual,
    visual_callback=data_recorder_visual,
    record_video=False
    )

mpc.print_timings()