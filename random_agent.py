import numpy as np
from maze2 import Maze


size = 9
max_steps = 30
num_episodes = 100
num_success = 0
env = Maze(size, max_steps)
for i in range(num_episodes):
    belief = np.ones((4, size, size)) / (size * size)
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    if reward:
        num_success += 1
print(num_success)
