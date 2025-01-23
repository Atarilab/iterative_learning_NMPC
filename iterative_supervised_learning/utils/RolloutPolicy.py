import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder  # type: ignore
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import load_mj_pin  # type: ignore

from mpc_controller.mpc import LocomotionMPC