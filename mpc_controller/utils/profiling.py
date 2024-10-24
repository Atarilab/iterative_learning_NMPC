import time
from functools import wraps
from typing import Dict, List
import numpy as np

def time_fn(method_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.compute_timings:
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                t = (end_time - start_time) * 1000 # ms
                # Update average
                self.timings[method_name].append(t)
            else:
                result = func(self, *args, **kwargs)
            return result
        return wrapper
    return decorator

def print_timings(timings : Dict[str, List[float]]):
    for fn_name, time_list in timings.items():
        print("---", fn_name, "---")
        mean = np.mean(time_list[1:])
        std = np.std(time_list[1:])
        max = np.max(time_list[1:])
        print(f"mean: {mean:.2f} ms")
        print(f"std: {std:.2f} ms")
        print(f"max: {max:.2f} ms")
        print(f"first: {time_list[0]:.2f} ms")