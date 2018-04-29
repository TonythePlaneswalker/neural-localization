from __future__ import print_function
import gym
import Maze
import getch
from visdom import Visdom
def main():
    vis = Visdom(server='http://localhost',port='8097')
    env = gym.make('Maze-v0')
    while True:
        env.render(vis)
        print('get key')
        key = getch.getch()
        print(key)
        if key == 'q':
            break
        elif key == 'w':
            action = 0
        elif key == 'a':
            action = 1
        elif key == 'd':
            action = 2
        else:
            continue
        status, reward, done, info = env.step(action)
        print('action:',action)
        print('reward:',reward)
        print('done:',done)
        if done:
            env.reset()



if __name__ == '__main__':
    main()
