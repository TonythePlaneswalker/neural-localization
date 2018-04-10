from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random_maze import maze
from ray_casting import ray_casting
import numpy as np
from gym import spaces
from gym.utils import seeding
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class Maze:
    '''
    Action: 0 Move Forward
    Action: 1 Turn Left 90 degrees
    Action: 2 Turn Right 90 degrees
    Orientation: 0 Facing North.
    Orientation: 1 Facing East
    Orientation: 2 Facing South
    Orientation: 3 Facing West
    '''
    def __init__(self, size, max_steps):
        self.size = size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=1, high=size, shape=(1,), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.seed()
        self.reset()

    def reset(self):
        self.map = ~maze(self.size, self.size, self.np_random)
        self.ray_cast = np.transpose(ray_casting(self.map), [2, 0, 1])
        self.orientation = self.np_random.randint(3)
        self.position = self.init_pos()
        self.belief = np.ones((4, self.size, self.size)) / (4 * self.size * self.size)
        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def init_pos(self):
        free_row, free_col = np.where(self.map)
        i = self.np_random.randint(len(free_row))
        return [free_row[i], free_col[i]]

    def step(self, action):
        if action == 1:
            self.orientation = (self.orientation - 1) % 4
            self.belief = np.roll(self.belief, -1, axis=0)
        elif action == 2:
            self.orientation = (self.orientation + 1) % 4
            self.belief = np.roll(self.belief, 1, axis=0)
        elif action == 0:
            new_pos = self.position.copy()
            if self.orientation == 0:
                new_pos[0] -= 1
            elif self.orientation == 1:
                new_pos[1] += 1
            elif self.orientation ==2:
                new_pos[0] += 1
            else:
                new_pos[1] -= 1
            if self.map[new_pos[0], new_pos[1]]:
                self.position = new_pos
            n = self.size
            for i in range(1, n - 1):
                self.belief[0, i, :] = [self.belief[0, i+1, j] if self.belief[0, i-1, j] else
                                        self.belief[0, i, j] + self.belief[0, i+1, j] for j in range(n)]
                self.belief[1, :, :] = [self.belief[1, j, i+1] if self.belief[1, j, i-1] else
                                        self.belief[1, j, i] + self.belief[1, j, i+1] for j in range(n)]
                self.belief[2, n-i-1, :] = [self.belief[2, n-i-2, j] if self.belief[2, n-i, j] else
                                        self.belief[2, n-i-2, j] + self.belief[2, n-i-1, j] for j in range(n)]
                self.belief[3, n-i-1, :] = [self.belief[3, j, i-2] if self.belief[3, j, n-i] else
                                        self.belief[3, j, i-2] + self.belief[3, j, i-1] for j in range(n)]

        obs = self.ray_cast[self.orientation, self.position[0], self.position[1]]
        lik = self.ray_cast == obs
        self.belief *= lik
        self.belief /= self.belief.sum()
        reward = self.belief.max()

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            if np.argmax(self.belief) == (self.position[0] * self.size + self.position[1]):  # Correctly localized
                reward = 1
            else:
                reward = 0
        else:
            done = False
        return [self.position,self.orientation], reward, done, {}
