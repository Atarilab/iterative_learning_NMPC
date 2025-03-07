import mujoco
import mujoco.viewer
import numpy as np
import time
from mj_pin.utils import get_robot_description, mj_frame_pos

# load robot model
robot_name = "go2"
robot_desc = get_robot_description(robot_name)
xml_path = robot_desc.xml_path
mj_model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(mj_model)

# body names and id
for i in range(mj_model.nbody):
    print(f"Body ID: {i}, Name: {mj_model.body(i).name}")

# joint names and id
for i in range(mj_model.njnt):
    print(f"Joint ID: {i}, Name: {mj_model.joint(i).name}")

# collision objects names and id
for i in range(mj_model.ngeom):
    print(f"Geom ID: {i}, Name: {mj_model.geom(i).name}")

# sites names and id (custom frames)
for i in range(mj_model.nsite):
    print(f"Site ID: {i}, Name: {mj_model.site(i).name}")

# define base name
frame_name = "base"
frame_id = mj_model.body(name = frame_name).id
# print(frame_id)

# Apply a force [Fx, Fy, Fz] and torque [Tx, Ty, Tz] to the base
# define force and torque
force = np.array([10.0, 0.0, 0.0])  # Example: 10N in the X direction
torque = np.array([0.0, 0.0, 0.0])  # No torque applied

# Set the force/torque at the base
data.xfrc_applied[frame_id, :3] = force  # Linear force
data.xfrc_applied[frame_id, 3:] = torque  # Rotational torque

# # Add a ground plane in Python (only for testing)
# floor_geom = mujoco.MjGeom()
# floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
# floor_geom.size = [10, 10, 0.1]
# floor_geom.pos = [0, 0, 0]
# mj_model.geom.append(floor_geom)

# Simulate and visualize
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        # apply force and torque
        data.xfrc_applied[frame_id, :3] = force  
        data.xfrc_applied[frame_id, 3:] = torque 
        mujoco.mj_step(mj_model, data)
        time.sleep(0.1)
        viewer.sync()

# NOTE: robot falls down because there is no floor