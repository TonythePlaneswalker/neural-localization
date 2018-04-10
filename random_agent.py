import numpy as np
from visdom import Visdom
import Maze
import gym

size = 9
max_steps = 30
num_episodes = 100
num_success = 0
env = gym.make('Maze-v0')
vis = Visdom(server='http://localhost',port='9000')

for i in range(num_episodes):


    belief = np.ones((4, size, size)) / (size * size)
    done = False
    state = env.reset()
    while not done:
        env.render(vis)
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    if reward:
        num_success += 1
print(num_success)
