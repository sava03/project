import numpy as np
from numpy import ndarray
from typing import Tuple

def maze_generator() -> ndarray:
    matrix: ndarray = np.array([
        [1]*10,
        [1,"S",1,0,0,0,1,0,0,1],
        [1,0,1,0,1,0,1,0,1,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,1,1,0,1,1,1,1,0,1],
        [1,0,1,"G",0,0,0,1,0,1],
        [1,0,0,0,1,1,0,0,0,1],
        [1,1,1,0,1,0,1,1,0,1],
        [1,0,0,0,0,0,1,"E",0,1],
        [1]*10
    ])
    
    return matrix


class State:
    def __init__(self, pos_x: int = 1, pos_y: int = 1, passed_sub_goal: bool = False):
        self.pos_x: int = pos_x
        self.pos_y: int = pos_y
        self.passed_sub_goal: bool = passed_sub_goal
        self.done: bool = False
        
    def update_pos(self, position: Tuple[int, int]):
        self.pos_x, self.pos_y = position
        

def reward_function(state: State, checkpoint_position: Tuple[int, int], end_position: Tuple[int, int]):
    if state.passed_sub_goal and (state.pos_x, state.pos_y) == end_position:
        state.done = True
        return 100
    elif not state.passed_sub_goal and (state.pos_x, state.pos_y) == checkpoint_position:
        state.passed_sub_goal = True
        return 10
    return -1
    
