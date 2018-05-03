from __future__ import print_function
import gym
import Maze
import getch
from visdom import Visdom


def main():
    vis = Visdom()
    env = gym.make('Maze-v3')
    env.generate_map()
    while True:
        done = False
        env.reset()
        while not done:
            env.render(vis)
            print(env.pos_cont)
            print(env.position)
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
            state, reward, done, info = env.step(action)
            print('action:', action)
            print('reward:', reward)
            print('done:', done)


if __name__ == '__main__':
    main()
