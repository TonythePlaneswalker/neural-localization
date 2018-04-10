import argparse
import Maze
import gym
import matplotlib.pyplot as plt
import os


def plot_env(env):
    map_fig = plt.figure()
    map_with_agent = env.get_map_with_agent().transpose([1, 2, 0])
    plt.imshow(map_with_agent)
    plt.axis('off')
    belief_fig = plt.figure(figsize=(10,3))
    orientations = ['North', 'East', 'South', 'West']
    for j in range(4):
        plt.subplot(1, 4, j + 1)
        plt.imshow(env.belief[j], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(orientations[j])
    return map_fig, belief_fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_size', type=int, default=7)
    parser.add_argument('--max_step', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_dir')
    args = parser.parse_args()

    env = gym.make('Maze-v1')
    env.set_size(args.map_size)
    env.set_max_step(args.max_step)
    env.generate_map()

    num_success = 0
    for i in range(args.num_episodes):
        done = False
        state = env.reset()
        step = 0
        while not done:
            if args.vis:
                env.render()
            if args.plot:
                map_fig, belief_fig = plot_env(env)
                map_fig.savefig(os.path.join(args.plot_dir, 'map_step%d' % step))
                belief_fig.savefig(os.path.join(args.plot_dir, 'belief_step%d' % step))
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            step += 1
        if args.plot:
            map_fig, belief_fig = plot_env(env)
            map_fig.savefig(os.path.join(args.plot_dir, 'map_step%d' % step))
            belief_fig.savefig(os.path.join(args.plot_dir, 'belief_step%d' % step))
        num_success += reward
    print('Success ratio', num_success / args.num_episodes)
