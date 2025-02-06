from mj_pin.utils import get_robot_description

from mpc_controller.mpc import LocomotionMPC
from search.data_utils import search_contact_plan, run_contact_plan, SearchConfig, RunConfig

Node = tuple[int, int, int, int]

ROBOT_NAME = "go2"
robot_description = get_robot_description(ROBOT_NAME)
feet_frames_mj = ["FL", "FR", "RL", "RR"]
feet_frames_pin = [foot + "_foot" for foot in feet_frames_mj]

sarch_cfg = SearchConfig(
    xml_path=robot_description.xml_scene_path,
    feet_frames_mj=feet_frames_mj,
    grid_size=(10, 10),
    spacing=(0.19, 0.19),
    size_ratio=(0.45, 0.45),
    randomize_pos_ratio=0.75,
    randomize_height_ratio=0.1,
    n_remove=0,
    height=0.2,
    shape="box",
    max_step_size=0.31,
    max_foot_displacement=0.265
)

run_cfg = RunConfig(
    xml_path=robot_description.xml_scene_path,
    n_cores=10,
    n_run=20,
    sim_dt=1.0e-3
)

stones, path, q0, v0 = search_contact_plan(sarch_cfg)

# New MPC instance
mpc = LocomotionMPC(
    path_urdf=robot_description.urdf_path,
    feet_frame_names=feet_frames_pin,
    robot_name=ROBOT_NAME,
    joint_ref = robot_description.q0,
    gait_name="slow_trot",
    contact_planner="custom",
    height_offset=stones.height,
    interactive_goal=False,
    sim_dt=run_cfg.sim_dt,
    print_info=False,
    )
mpc.contact_planner.set_contact_locations(stones.positions[path].copy())
sim_time = (len(path) + 3) * mpc.config_gait.nominal_period

# Run sim
success = run_contact_plan(mpc, sim_time, stones, path, q0, v0, run_cfg, use_viewer=True)