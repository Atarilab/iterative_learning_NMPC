import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple

@dataclass
class ContactMode:
    cnt_pos: List[np.ndarray] = field(default_factory=list)
    eeff_in_cnt: List[int] = field(default_factory=list)

    def __post_init__(self):
        """
        Post-initialization to check the integrity of cnt_pos and eeff_id.
        """
        n_cnt = len(self.cnt_pos)
        n_id = len(self.eeff_in_cnt)
        assert n_cnt == n_id, (
            f"The number of contact locations ({n_cnt}) and end-effector indices ({n_id}) must be equal."
        )

        # Ensure that all positions are 3D points
        for pos in self.cnt_pos:
            assert pos.shape == (3,), "Each contact position must be a 3D point (shape (3,))."

class ContactPlan:
    def __init__(self, eeff_names: List[str]) -> None:
        self.eeff_names = eeff_names
        self.n_eeff = len(eeff_names)
        self.eeff_name2id = {name: i for i, name in enumerate(eeff_names)}
        self.eeff_id2name = {i: name for i, name in enumerate(eeff_names)}

        # Plan contact modes
        self.modes: List[ContactMode] = []
        self.i_next_mode: int = 0

    def add(self, cnt_pos: List[np.ndarray], eeff_name: List[str]) -> None:
        """
        Add a contact mode to the plan.
        Args:
            cnt_pos (List[np.ndarray]): Contact positions. List of 3D points.
            eeff_name (List[str]): Names of the end-effectors in contact for the given contact mode.
        """
        eeff_id = [self.eeff_name2id[name] for name in eeff_name]
        next_cnt_mode = ContactMode(cnt_pos, eeff_id)
        self.modes.append(next_cnt_mode)

    def _update_next_mode(self, eeff_in_cnt: List[str]) -> None:
        """
        Update the next mode if <eeff_in_cnt> matches the current contact mode.
        """
        if self.i_next_mode < len(self.modes):
            current_mode_set = set(self.modes[self.i_next_mode].eeff_in_cnt)
            eeff_in_cnt_set = set(self.eeff_name2id[name] for name in eeff_in_cnt)

            if current_mode_set == eeff_in_cnt_set:
                self.i_next_mode += 1
        else:
            print("Last contact mode reached.")

    def next(self, current_eeff_in_cnt: List[str], n: int = 1) -> List[ContactMode]:
        """
        Return the <n> next contact modes based on the current end-effectors in contact.
        """
        self._update_next_mode(current_eeff_in_cnt)

        # Get the next <n> contact modes and repeat the last mode if necessary
        next_cnt_modes = self.modes[self.i_next_mode:self.i_next_mode + n]

        # Repeat the last contact mode if there are not enough remaining modes
        repeat_last = n - len(next_cnt_modes)
        if repeat_last > 0 and next_cnt_modes:
            next_cnt_modes += [next_cnt_modes[-1]] * repeat_last

        return next_cnt_modes


class CyclicContactPlan(ContactPlan):
    def __init__(self, eeff_names: List[str]) -> None:
        super().__init__(eeff_names)

    def _update_next_mode(self, eeff_in_cnt: List[str]) -> None:
        """
        Update the next mode and loop back to the beginning if end is reached.
        """
        if self.i_next_mode >= len(self.modes):
            self.i_next_mode = 0
        
        current_mode_set = set(self.modes[self.i_next_mode].eeff_in_cnt)
        eeff_in_cnt_set = set(self.eeff_name2id[name] for name in eeff_in_cnt)

        if current_mode_set == eeff_in_cnt_set:
            self.i_next_mode += 1
            if self.i_next_mode >= len(self.modes):
                self.i_next_mode = 0  # Loop back to the first mode

    def next(self, current_eeff_in_cnt: List[str], n: int = 1) -> List[ContactMode]:
        """
        Return the <n> next contact modes, cycling back to the first mode if necessary.
        """
        self._update_next_mode(current_eeff_in_cnt)
        n_modes = len(self.modes)

        next_cnt_modes = [
            self.modes[(self.i_next_mode + i) % n_modes]
            for i in range(n)
        ]

        return next_cnt_modes
