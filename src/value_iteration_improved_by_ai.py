from basics import maze_generator, reward_function
from numpy import ndarray
import numpy as np
from typing import Dict, Callable, Tuple


def value_iteration(maze                : ndarray           = maze_generator(), 
                    reward_function     : Callable          = reward_function,
                    threshold           : float             = 0.5, 
                    discount_value      : float             = 0.9,
                    checkpoint_position : Tuple[int, int]   = (3,5), 
                    end_position        : Tuple[int, int]   = (7,8)
                    ) -> ndarray:
    
    value_func  : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    policy      : Dict[bool, ndarray] = {True: np.zeros(shape=maze.shape), False: np.zeros(shape=maze.shape)}
    
    
    biggest_change      : float             = float('Inf')
    passed_checkpoint   : Tuple[bool, bool] = (True, False)
    
    while biggest_change < threshold:
        biggest_change: float = 0.0
        
        for checkpoint_value in passed_checkpoint:         #   \
            for x in range(maze.shape[0]):                 #   |---- Go over all possible states
                for y in range(maze.shape[1]):             #   /
                    
                    if maze[x,y] == 1:              # walls cannot be reached so there is no point in going over those
                        continue
                
                    current_pos : Tuple[int,int]    = (x,y)
                    best_value  : float             = -float('inf')
                    best_move   : str               = None
                    
                    for move in ['up', 'down', 'left', 'right']:
                        
                        new_pos = (current_pos[0] + {'up': -1, 'down': 1, 'left': 0, 'right': 0}[move],     #   \ Go over all moves
                                    current_pos[1] + {'up': 0, 'down': 0, 'left': -1, 'right': 1}[move])    #   / that are feasible in a state
                        
                        if maze[new_pos[0], new_pos[1]] == 1:
                            new_pos = current_pos
                            
                        value = reward_function(agent_position=new_pos, passed_checkpoint=checkpoint_value, checkpoint_position=checkpoint_position, end_position=end_position) \
                            + discount_value * value_func[checkpoint_value][new_pos]
                    
                        if value > best_value:
                            best_value = value
                            best_move = move
                        
                    old_value   : float = value_func[checkpoint_value][current_pos]
                    change      : float = best_value - old_value
                    
                    if change > biggest_change:
                        biggest_change = change
                        
                    value_func[checkpoint_value][current_pos] = best_value
                    policy[checkpoint_value][current_pos] = best_move
                        
    return policy

print(value_iteration())