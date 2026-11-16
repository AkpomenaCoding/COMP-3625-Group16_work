"""
Grid world class. Run this file for an interactive demo
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple

class GridWorld:

    def __init__(self, 
                 size=5, 
                 wall_probability=0.2, 
                 hazard_probability=0.1, 
                 state_includes_location=True, 
                 state_includes_sensory=False, 
                 randomize_each_trial=False,
                 ):
        """
        generates a grid world environment
        :param size: side length of the (square) gridworld
        :param wall_probability: probability of a cell being a wall
        :param hazard_probability: probability of a cell being a hazard
        :param state_includes_location: if True, the state dict contains the agent's x/y coordinates
        :param state_includes_sensory: if True, the state dict contains information about neighboring cells
        :param randomize_each_trial: if True, the gridworld is re-randomized each time .reset() is called, and when the agent reaches the goal.
        """
        if wall_probability + hazard_probability >= 1.0:
            raise ValueError('wall_fraction + hazard_fraction must be less than 1.0')
        
        if not state_includes_location and not state_includes_sensory:
            raise ValueError("must use at least one of state_includes_location and state_includes_sensory")

        self.wall_probability = wall_probability
        self.hazard_probability = hazard_probability
        self.randomize_each_trial = randomize_each_trial
        self.size = size
        

        self.colors = {
            'wall': (0.5, 0.5, 0.5),
            'hazard': (1.0, 0.0, 0.0),
            'goal': (0.0, 1.0, 0.0)
        }

        self.current_coords = (0, 0)
        self.state_includes_location=state_includes_location 
        self.state_includes_sensory=state_includes_sensory

        self.directions = {
            'up': np.array([-1, 0]), 
            'right': np.array([0, 1]), 
            'down': np.array([1, 0]), 
            'left': np.array([0, -1]),
        }

        self._generate_grid()
        
    def _generate_grid(self):
        self.grid = np.array([[None] * self.size] * self.size)
        start = 0, 0
        goal = self.size-1, self.size-1

        # guarantee a clear path to the goal
        path = [start]
        while path[-1] != goal:
            if random.random() < 0.5:
                path.append((path[-1][0] + 1, path[-1][1]))
            else:
                path.append((path[-1][0], path[-1][1] + 1))
            if path[-1][0] >= self.size or path[-1][1] >= self.size:
                del path[-1]
        path = set(path)
        
        # select walls/hazards from remaining cells
        empty = {(i, j) for i in range(self.size) for j in range(self.size)}
        empty -= path
        num_empty = len(empty)
        walls = set(random.sample(sorted(empty), k=int(num_empty * self.wall_probability)))
        empty -= walls
        hazards = set(random.sample(sorted(empty), k=int(num_empty * self.hazard_probability)))

        for wall in walls:
            self.grid[wall[0], wall[1]] = 'wall'
        for hazard in hazards:
            self.grid[hazard[0], hazard[1]] = 'hazard'
        self.grid[self.size-1, self.size-1] = 'goal'

    def get_current_state(self):
        state = {}
        if self.state_includes_location:
            state['x'] = int(self.current_coords[1])
            state['y'] = int(self.current_coords[0])

        if self.state_includes_sensory:
            for direction, state_delta in self.directions.items():
                test_pt = state_delta + self.current_coords
                for obj_type in ['wall', 'hazard', 'goal']:
                    state[obj_type + '_' + direction] = 0
                
                if test_pt.min() < 0 or test_pt.max() >= self.size or self.grid[test_pt[0], test_pt[1]] == 'wall':
                    state['wall_' + direction] = 1

                elif self.grid[test_pt[0], test_pt[1]] is not None:
                    state[self.grid[test_pt[0], test_pt[1]] + '_' + direction] = 1
        return state

    def get_actions(self, state=None):
        return list(self.directions)

    def reset(self) -> dict:
        if self.randomize_each_trial:
            self._generate_grid()
        self.current_coords = (0, 0)
        return self.get_current_state()

    def render(self):

        img = np.ones((self.size, self.size, 3))
        for i in range(self.size):
            for j in range(self.size):
                img[i, j] = self.colors.get(self.grid[i, j], (1.0, 1.0, 1.0))
        
        plt.cla()
        plt.imshow(img)
        plt.text(x=self.current_coords[1], y=(self.current_coords[0]), s='A', horizontalalignment='center', verticalalignment='center', size='xx-large')
        plt.pause(0.001)
        

    def step(self, action: str) -> tuple[dict, float]:
        """
        execute one step in the environment
        :param action: action to perform: 'up', 'down', 'left', or 'right'
        :return: new state, reward
        """
        new_coords = self.current_coords + self.directions[action]
        if not(new_coords.min() < 0 or new_coords.max() >= self.size or self.grid[new_coords[0], new_coords[1]] == 'wall'):
            self.current_coords = tuple(new_coords)
        
        reward = 0
        if self.grid[self.current_coords[0], self.current_coords[1]] == 'hazard':
            reward = -1
        elif self.grid[self.current_coords[0], self.current_coords[1]] == 'goal':
            reward = 1
            self.reset()
        
        return self.get_current_state(), reward




if __name__ == '__main__':
    gw = GridWorld(size=5, 
                   wall_probability=0.2, 
                   hazard_probability=0.1, 
                   state_includes_sensory=True, 
                   randomize_each_trial=True,
                   )
    state = gw.reset()


    while True:
        gw.render()    
        print('current state:', gw.get_current_state())
        action = input(f'select action from {gw.get_actions()} or "q" to quit: ')
        if action.lower() == 'q':
            exit()

        new_state, reward = gw.step(action)
        print('reward:', reward)
        