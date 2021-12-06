from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
import gym

from common import callbacks

GymEnv = Union[gym.Env]
Schedule = Callable[[float], float]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]