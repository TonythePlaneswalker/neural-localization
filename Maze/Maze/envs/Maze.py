from __future__ import print_function
import random_maze
import ray_casting
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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
    def __init__(self, size=17):
        self.size = size
        self.max_steps = 800
        self.current_step = 0
        self.observation_space = spaces.Box(low=1,high=self.size,shape
                                            =(1,),dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.map = random_maze.maze(self.size,self.size,np.random)
        self.ray_cast = ray_casting.ray_casting(self.map)
        self.orientation = np.random.randint(low=0,high=3)
        self.possible_position = self.findOpenSpace()
        self.position = self.init_pos()
        self.possibleMap = np.ones((self.size,self.size,4))
        self.viewer = None

    def reset(self):
        self.orientation = np.random.randint(low=0,high=3)
        self.possible_position = self.findOpenSpace()
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
        return self.ray_cast[self.position[0],self.position[1],(self.orientation-1)%4]

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
                self.position[1] -= 1
            elif self.orientation == 1:
                self.position[0] -= 1
            elif self.orientation ==2:
                self.position[1] += 1
            else:
                self.position[0] += 1

            if self.map[self.position[0],self.position[1]]:
                # Crushed!
                print("Crushed!")
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
        if done:
            print("Find!")
        return [self.position,self.orientation], reward, done, {}

    def render(self,vis):
        # img = self._get_image()
        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.SimpleImageViewer()
        # self.viewer.imshow(img)
        # return self.viewer.isopen
        tmp = np.array([~self.map,~self.map,~self.map]).astype(np.float32)

        tmp = np.kron(tmp,np.ones((20,20)))
        tmp = np.transpose(tmp,(1,2,0)).copy()
        center_point = np.array([self.position[1]*20+9,self.position[0]*20+9])
        if self.orientation == 1:
            pts = np.array([[center_point[0],center_point[1]-6],[center_point[0]-3*np.sqrt(3),center_point[1]+3],
                [center_point[0]+3*np.sqrt(3),center_point[1]+3]])
        elif self.orientation == 2:
            pts = np.array([[center_point[0]+6,center_point[1]],[center_point[0]-3,center_point[1]-3*np.sqrt(3)],
                [center_point[0]-3,center_point[1]+3*np.sqrt(3)]])
        elif self.orientation == 3:
            pts = np.array([[center_point[0],center_point[1]+6],[center_point[0]-3*np.sqrt(3),center_point[1]-3],
                [center_point[0]+3*np.sqrt(3),center_point[1]-3]])
        else:
            pts = np.array([[center_point[0]-6,center_point[1]],[center_point[0]+3,center_point[1]-3*np.sqrt(3)],
                [center_point[0]+3,center_point[1]+3*np.sqrt(3)]])
        pts = pts.reshape((-1,1,2)).astype(np.int32)
        cv2.fillConvexPoly(tmp,pts,1)
        tmp = np.transpose(tmp,(2,0,1))
        tmp_n = np.array([self.possibleMap[:,:,0],self.possibleMap[:,:,0],self.possibleMap[:,:,0]]).astype(np.float32)
        tmp_e = np.array([self.possibleMap[:,:,1],self.possibleMap[:,:,1],self.possibleMap[:,:,1]]).astype(np.float32)
        tmp_s = np.array([self.possibleMap[:,:,2],self.possibleMap[:,:,2],self.possibleMap[:,:,2]]).astype(np.float32)
        tmp_w = np.array([self.possibleMap[:,:,3],self.possibleMap[:,:,3],self.possibleMap[:,:,3]]).astype(np.float32)
        tmp_n = np.kron(tmp_n,np.ones((20,20)))
        tmp_e = np.kron(tmp_e,np.ones((20,20)))
        tmp_s = np.kron(tmp_s,np.ones((20,20)))
        tmp_w = np.kron(tmp_w,np.ones((20,20)))
        vis.image(tmp,win='map',opts={'title':"Maze"})
        vis.image(tmp_n,win='North',opts={'title':"North"})
        vis.image(tmp_e,win='East',opts={'title':"East"})
        vis.image(tmp_s,win='South',opts={'title':"South"})
        vis.image(tmp_w,win='West',opts={'title':"West"})


    def _get_image(self, scale=20):
        image_arr = self.map.copy().astype(np.uint8)
        image_arr[self.map == 1] = 255
        #image_arr[~self.observed] = 127
        image_arr = np.kron(image_arr, np.ones((scale, scale), dtype=np.uint8))
        image_arr = np.stack([image_arr for i in range(3)], axis=2)
        image = Image.fromarray(image_arr)

        draw = ImageDraw.Draw(image)
        #x0, y0, x1, y1 = np.array(self.possible_position) * scale
        #draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0))
        x0, y0, x1, y1 = np.array([self.position[1], self.position[0], self.position[1] + 1, self.position[0] + 1]) * scale
        # Draws a 1/4 circle opposite to the agent's orientation, which resembles an arrow
        # Angles are measured from 3 o'clock, increasing clockwise.
        draw.pieslice([(x0, y0), (x1, y1)],
                      45 + 90 * self.orientation,
                      45 + 90 * (self.orientation + 1),
                      fill=(255, 0, 0))
        del draw

        image_arr = np.array(image.getdata()).reshape(image_arr.shape).astype(np.uint8)
        return image_arr
