from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import gym
import numpy as np
from . import random_maze, ray_casting
from gym import spaces
from gym.utils import seeding
from visdom import Visdom


class maze(gym.Env):
    '''
    Action: 0 Move Forward
    Action: 1 Turn Left 90 degrees
    Action: 2 Turn Right 90 degrees
    Orientation: 0 Facing North.
    Orientation: 1 Facing East
    Orientation: 2 Facing South
    Orientation: 3 Facing West
    '''

    def __init__(self):
        self.size = 9
        self.max_step = 15
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.vis = Visdom()
        self.seed()

    def set_size(self, size):
        self.size = size

    def set_max_step(self, max_step):
        self.max_step = max_step

    def generate_map(self):
        self.map = ~random_maze.maze(self.size, self.size, self.np_random, complexity=0.99, density=0.99)
        self.ray_cast = np.transpose(ray_casting.ray_cast(self.map), [2, 0, 1])

    def reset(self):
        self.orientation = self.np_random.randint(3)
        self.position = self.init_pos()
        init_obs = self.ray_cast[self.orientation, self.position[0], self.position[1]]
        self.belief = (self.ray_cast == init_obs).astype(np.float32)
        self.belief /= self.belief.sum()
        self.current_step = 0
        return self.belief

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
            new_pos = self.position[:]
            tmp_pos = self.position[:]
            epsilon = np.random.choice(3,p=[0.10558,0.78884,0.10558]) - 1
            if self.orientation == 0:
                new_pos[0] -= 1+epsilon
                tmp_pos[0] -= 1
            elif self.orientation == 1:
                new_pos[1] += 1+epsilon
                tmp_pos[1] += 1
            elif self.orientation == 2:
                new_pos[0] += 1+epsilon
                tmp_pos[0] += 1
            else:
                new_pos[1] -= 1+epsilon
                tmp_pos[1] -= 1
            if new_pos[0] >=0 and new_pos[0] < self.size and new_pos[1] >= 0 and new_pos[1] < self.size:
                if self.map[new_pos[0], new_pos[1]] and (epsilon != 1 or self.map[tmp_pos[0],tmp_pos[1]]):
                    self.position = new_pos
            elif epsilon == 1 and self.map[tmp_pos[0], tmp_pos[1]]:
                self.position = tmp_pos

            n = self.size
            belief_tmp = self.belief.copy()
            for i in range(1,n-1):
                for j in range(n):
                    if self.map[i-1,j] and i+2 < n:
                        belief_tmp[0,i,j] = 0.78884*self.belief[0,i+1,j] + 0.10558*self.belief[0,i,j] \
                                            + 0.10558*self.belief[0,i+2,j]
                    elif self.map[i-1,j]:
                        belief_tmp[0,i,j] = 0.882*self.belief[0,i+1,j] + 0.118*self.belief[0,i,j]
                    elif self.map[i-1,j] == 0 and i+2 < n:
                        belief_tmp[0,i,j] = 0.7884*(self.belief[0,i+1,j]+self.belief[0,i,j]) + \
                                            + 0.10558*(self.belief[0,i,j]) + \
                                            0.10558*(self.belief[0,i,j]+self.belief[0,i+1,j]+self.belief[0,i+2,j])
                    elif self.map[i-1,j] == 0:
                        belief_tmp[0,i,j] = 0.882*(self.belief[0,i+1,j]+self.belief[0,i,j]) + \
                                            + 0.118*(self.belief[0,i,j])
                    if self.map[j,n-i] and i+2 < n:
                        belief_tmp[1,j,n-i-1] = 0.78884*self.belief[1,j,n-i-2] + 0.10558*self.belief[1,j,n-i-1] \
                                            + 0.10558*self.belief[1,j,n-i-3]
                    elif self.map[j,n-i]:
                        belief_tmp[1,j,n-i-1] = 0.882*self.belief[1,j,n-i-2] + 0.118*self.belief[1,j,n-i-1]
                    elif self.map[j,n-i]==0 and i+2 < n:
                        belief_tmp[1,j,n-i-1] = 0.7884*(self.belief[1,j,n-i-2]+self.belief[1,j,n-i-1]) + \
                                            + 0.10558*(self.belief[1,j,n-i-1]) + \
                                            0.10558*(self.belief[1,j,n-i-2]+self.belief[1,j,n-i-1]+self.belief[1,j,n-i-3])
                    elif self.map[j,n-i]==0:
                        belief_tmp[1,j,n-i-1] = 0.882*(self.belief[1,j,n-i-2]+self.belief[1,j,n-i-1]) + \
                                            + 0.118*(self.belief[1,j,n-i-1])
                    if self.map[n-i,j] and i+2 < n:
                        belief_tmp[2,n-i-1,j] = 0.78884*self.belief[2,n-i-2,j] + 0.10558*self.belief[2,n-i-1,j] \
                                            + 0.10558*self.belief[2,n-i-3,j]
                    elif self.map[n-i,j]:
                        belief_tmp[2,n-i-1,j] = 0.882*self.belief[2,n-i-2,j] + 0.118*self.belief[2,n-i-1,j]
                    elif self.map[n-i,j]==0 and i+2 < n:
                        belief_tmp[2,n-i-1,j] = 0.7884*(self.belief[2,n-i-2,j]+self.belief[2,n-i-1,j]) + \
                                            + 0.10558*(self.belief[2,n-i-1,j]) + \
                                            0.10558*(self.belief[2,n-i-2,j]+self.belief[2,n-i-1,j]+self.belief[2,n-i-3,j])
                    elif self.map[n-i,j]==0:
                        belief_tmp[2,n-i-1,j] = 0.882*(self.belief[2,n-i-2,j]+self.belief[2,n-i-1,j]) + \
                                            + 0.118*(self.belief[2,n-i-1,j])
                    if self.map[j,i-1] and i+2 < n:
                        belief_tmp[3,j,i] = 0.78884*self.belief[3,j,i+1] + 0.10558*self.belief[3,j,i] \
                                            + 0.10558*self.belief[3,j,i+2]
                    elif self.map[j,i-1]:
                        belief_tmp[3,j,i] = 0.882*self.belief[3,j,i+1] + 0.118*self.belief[3,j,i]
                    elif self.map[j,i-1] == 0 and i+2 < n:
                        belief_tmp[3,j,i] = 0.7884*(self.belief[3,j,i+1]+self.belief[3,j,i]) + \
                                            + 0.10558*(self.belief[3,j,i]) + \
                                            0.10558*(self.belief[3,j,i]+self.belief[3,j,i+1]+self.belief[3,j,i+2])
                    elif self.map[j,i-1] ==0:
                        belief_tmp[3,j,i] = 0.882*(self.belief[3,j,i+1]+self.belief[3,j,i]) + \
                                            + 0.118*(self.belief[3,j,i])
            self.belief = belief_tmp

        #print(self.belief)
        #_ = raw_input()
        obs = self.ray_cast[self.orientation, self.position[0], self.position[1]]
        # print(self.position, self.orientation, obs)
        epsilon = np.random.choice(3,p=[0.10558,0.78884,0.10558]) - 1
        obs = obs + epsilon
        if obs == -1:
            lik = (self.ray_cast == 0) * 0.882 + (self.ray_cast == 1) * 0.118
        else:
            lik = (self.ray_cast == (obs-1))*0.10558 + (self.ray_cast == (obs+1))*0.10558 + (self.ray_cast == obs)*0.7884
        #lik = self.ray_cast == obs
        self.belief *= lik
        self.belief /= self.belief.sum()
        reward = self.belief.max()

        self.current_step += 1
        if self.current_step >= self.max_step:
            done = True
            num_possible = np.sum(self.belief > 0)
            max_value = np.max(self.belief)
            o, y, x = np.unravel_index(np.argmax(self.belief), self.belief.shape)
            # print("belief:", o,y,x)
            # print("real:",self.orientation,self.position[0],self.position[1])
            if max_value > 0.5 and o == self.orientation and y == self.position[0] and x == self.position[1]:
                reward = 1
            else:
                reward = 0
        else:
            done = False
        return self.belief, reward, done, {}

    def get_map_with_agent(self):
        img = np.array([self.map, self.map, self.map]).astype(np.float32)
        img = np.kron(img,np.ones((20,20)))
        img = np.transpose(img,(1,2,0)).copy()
        center_point = np.array([self.position[1]*20+9,self.position[0]*20+9])
        if self.orientation == 0:
            pts = np.array([[center_point[0],center_point[1]-6],[center_point[0]-3*np.sqrt(3),center_point[1]+3],
                [center_point[0]+3*np.sqrt(3),center_point[1]+3]])
        elif self.orientation == 1:
            pts = np.array([[center_point[0]+6,center_point[1]],[center_point[0]-3,center_point[1]-3*np.sqrt(3)],
                [center_point[0]-3,center_point[1]+3*np.sqrt(3)]])
        elif self.orientation == 2:
            pts = np.array([[center_point[0],center_point[1]+6],[center_point[0]-3*np.sqrt(3),center_point[1]-3],
                [center_point[0]+3*np.sqrt(3),center_point[1]-3]])
        else:
            pts = np.array([[center_point[0]-6,center_point[1]],[center_point[0]+3,center_point[1]-3*np.sqrt(3)],
                [center_point[0]+3,center_point[1]+3*np.sqrt(3)]])
        pts = pts.reshape((-1,1,2)).astype(np.int32)
        cv2.fillConvexPoly(img,pts,1)
        img = np.transpose(img,(2,0,1))
        return img

    def render(self, mode='human'):
        self.vis.image(self.get_map_with_agent(), env='render', win='map', opts={'title': "Maze"})
        self.vis.image(np.kron(self.belief[0, :, :], np.ones((20, 20))),
                       env='render', win='North', opts={'title': "North"})
        self.vis.image(np.kron(self.belief[1, :, :], np.ones((20, 20))),
                       env='render', win='East', opts={'title': "East"})
        self.vis.image(np.kron(self.belief[2, :, :], np.ones((20, 20))),
                       env='render', win='South', opts={'title': "South"})
        self.vis.image(np.kron(self.belief[3, :, :], np.ones((20, 20))),
                       env='render', win='West', opts={'title': "West"})
