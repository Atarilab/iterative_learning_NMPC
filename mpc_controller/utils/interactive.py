import numpy as np
from typing import Tuple
from mj_pin.abstract import Keyboard

class SetVelocityGoal(Keyboard):
    VX_STEP = 0.05
    VY_STEP = 0.035
    VX_BOUND = [-1., 1.5]
    VY_BOUND = [-1., 1.]
    YAW_STEP = 0.2
    YAW_BOUND = [-1., 1.]

    def __init__(self):
        # Linear velocity
        self.v = np.zeros(3)
        # Angular velocity z
        self.yaw = 0.
        super().__init__()

    def on_key(self, **kwargs):
        if self.last_key == "w":
            self.v[0] += SetVelocityGoal.VX_STEP
            self.v[0] = min(self.v[0], SetVelocityGoal.VX_BOUND[1])
        elif self.last_key == "s":
            self.v[0] -= SetVelocityGoal.VX_STEP
            self.v[0] = max(self.v[0], SetVelocityGoal.VX_BOUND[0])
        elif self.last_key == "a":
            self.v[1] += SetVelocityGoal.VY_STEP
            self.v[1] = min(self.v[1], SetVelocityGoal.VY_BOUND[1])
        elif self.last_key == "d":
            self.v[1] -= SetVelocityGoal.VY_STEP
            self.v[1] = max(self.v[1], SetVelocityGoal.VY_BOUND[0])
        elif self.last_key == "q":
            self.yaw += SetVelocityGoal.YAW_STEP
            self.yaw = min(self.yaw, SetVelocityGoal.YAW_BOUND[1])
        elif self.last_key == "e":
            self.yaw -= SetVelocityGoal.YAW_STEP
            self.yaw = max(self.yaw, SetVelocityGoal.YAW_BOUND[0])
        elif self.last_key == " ":
            self.v = np.zeros(3)
            self.yaw = 0.
            
        self.v = np.round(self.v, 3)
        self.yaw = np.round(self.yaw, 3)

    def get_velocity(self) -> Tuple[np.ndarray, float]:
        v, yaw = self.v.copy(), self.yaw
        return v, yaw