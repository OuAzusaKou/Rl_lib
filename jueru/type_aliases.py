from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
import gym



GymEnv = Union[gym.Env]
Schedule = Callable[[float], float]