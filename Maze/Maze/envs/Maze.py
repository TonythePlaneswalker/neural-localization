from __future__ import print_function
import random_maze
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

Class Maze(gym.Env):
    '''
    Action: 0 Move Forward
    Action: 1 Turn Left 90 degrees
    Action: 2 Turn Right 90 degrees
    Orientation: 0 Facing North.
    Orientation: 1 Facing East
    Orientation: 2 Facing South
    Orientation: 3 Facing West
    '''
    def __init__(self, size=17):
        self.size = size
        self.max_steps = 800
        self.current_step = 0
        self.observation_space = spaces.Box(low=1,high=self.size,shape
                                            =(1,),dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.map = random_maze.generate_map(self.size)
        self.ray_cast = generate_ray_cast(self.map)
        self.orientation = np.random.randint(low=0,high=3)
        self.possible_position = self.findOpenSpace()
        self.position = self.init_pos()
        self.possibleMap = np.ones((self.size,self.size,4))

    def reset(self):
        self.orientation = np.random.randint(low=0,high=3)
        self.position = self.init_pos()
        self.possibleMap = np.ones((self.size,self.size,4))
        self.current_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def findOpenSpace(self):
        spaceList=[]
        for i in range(self.size):
            for j in range(self.size):
                if not self.map[i,j]:
                    spaceList.append([i,j])
        return spaceList

    def init_pos(self):
        return self.possible_position[np.random.randint(low=0,
                                        high=len(self.possible_position)-1)]

    def updatePossibleMap(self,observation):
        cur_map = np.zeros((self.size,self.size,4))
        for i in range(self.size):
            for j in range(self.size):
                for k in range(4):
                    if self.ray_cast[i,j,k] == observation:
                        cur_map[i,j,k] = 1
        self.possibleMap = np.multiply(self.possibleMap,cur_map)

    def getObservation(self):
        return self.ray_cast[self.position[0],self.position[1],self.orientation]

    def step(self, action):
        self.current_step += 1
        if action == 1:
            self.orientation = (self.orientation-1) % 4
            self.possibleMap = np.roll(self.possibleMap,-1,axis=2)
        elif action == 2:
            self.orientation = (self.orientation+1) % 4
            self.possibleMap = np.roll(self.possibleMap,1,axis=2)
        elif action == 0:
            if self.orientation == 0:
                self.position[0] -= 1
            elif self.orientation == 1:
                self.position[1] += 1
            elif self.orientation ==2:
                self.position[0] += 1
            else:
                self.position[1] -= 1
            if self.position not in self.possible_position:
                # Crushed!
                done = True
                reward = 0
                return [self.position,self.orientation], reward, done, {}
            else:
                self.possibleMap[:,:,0] = np.roll(self.possibleMap[:,:,0],-1,axis=0)
                self.possibleMap[:,:,1] = np.roll(self.possibleMap[:,:,1],1,axis=1)
                self.possibleMap[:,:,2] = np.roll(self.possibleMap[:,:,2],1,axis=0)
                self.possibleMap[:,:,3] = np.roll(self.possibleMap[:,:,3],-1,axis=1)

        obs = self.getObservation()
        self.updatePossibleMap(obs)
        n_possible = np.sum(self.possibleMap)
        reward = 1.0/n_possible
        done = True if n_possible == 1 else False
        return [self.position,self.orientation], reward, done, {}

    def render(self,mode='human'):
        img = self._get_image()
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen
    def _get_image(self, scale=20):
        image_arr = self.map.copy().astype(np.uint8)
        image_arr[self.map == 1] = 255
        image_arr[~self.observed] = 127
        image_arr = np.kron(image_arr, np.ones((scale, scale), dtype=np.uint8))
        image_arr = np.stack([image_arr for i in range(3)], axis=2)
        image = Image.fromarray(image_arr)

        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = np.array(self._get_obs_rect()) * scale
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0))
        x0, y0, x1, y1 = np.array([self.pos[1], self.pos[0], self.pos[1] + 1, self.pos[0] + 1]) * scale
        # Draws a 1/4 circle opposite to the agent's orientation, which resembles an arrow
        # Angles are measured from 3 o'clock, increasing clockwise.
        draw.pieslice([(x0, y0), (x1, y1)],
                      45 + 90 * self.dir,
                      45 + 90 * (self.dir + 1),
                      fill=(255, 0, 0))
        del draw

        image_arr = np.array(image.getdata()).reshape(image_arr.shape).astype(np.uint8)
        return image_arr
