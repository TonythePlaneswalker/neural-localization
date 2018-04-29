import argparse
import gym
import Maze
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from a2c_v2 import A2C


def plot_env(env, step, save_path):
    fig = plt.figure(figsize=(10, 8))
    grid = gridspec.GridSpec(2, 1, left=0.02, right=0.98, bottom=0., top=0.95, hspace=0.1, height_ratios=[0.5, 0.3])
    map_img = env.get_map_with_agent().transpose([1, 2, 0])
    ax = plt.Subplot(fig, grid[0])
    ax.imshow(map_img)
    ax.set_axis_off()
    ax.set_title('Step %d' % step)
    fig.add_subplot(ax)

    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[1], wspace=0.05)
    orientations = ['North', 'East', 'South', 'West']
    for j in range(4):
        ax = plt.Subplot(fig, inner_grid[j])
        ax.imshow(env.belief[j], cmap='gray')
        ax.set_axis_off()
        ax.set_title(orientations[j])
        fig.add_subplot(ax)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=15, help="Size of the random maze.")
    parser.add_argument('--max_step', type=int, default=20, help="Maximum number of steps in an episode.")
    parser.add_argument('--num_episodes', type=int, default=1, help="Number of test episodes.")
    parser.add_argument('--model_path', help="Path of the saved model weights.")
    parser.add_argument('-n', type=int, default=20, help="Number steps in the trace.")
    parser.add_argument('--stochastic', action='store_true', help="Use stochastic policy in testing.")
    parser.add_argument('--passive', action='store_true', help="Use a passive agent (random policy).")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_dir')
    args = parser.parse_args()

    env = gym.make('Maze-v1')
    env.set_size(args.size)
    env.set_max_step(args.max_step)

    passive = True if args.passive else False
    if not passive:
        agent = A2C(env, 0, args.n)
        state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        agent.model.load_state_dict(state_dict)
        stochastic = True if args.stochastic else False

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    for i in range(args.num_episodes):
        env.seed(i)
        env.generate_map()
        done = False
        belief = env.reset()
        history = np.zeros(5, dtype=np.int64)
        step = 0
        while not done:
            if args.render:
                env.render()
            if args.plot:
                plot_env(env, step, os.path.join(args.plot_dir, 'step%d' % step))
            if passive:
                action = env.action_space.sample()
            else:
                action, _, _, _ = agent.select_action(belief, env.map, history, step, stochastic)
            belief, reward, done, _ = env.step(action)
            history = np.roll(history, 1)
            history[0] = action + 1
            step += 1
        if args.plot:
            plot_env(env, step, os.path.join(args.plot_dir, 'step%d' % step))
        num_success += reward
    end = time.time()
    print('Success ratio %.3f time %.2f' % (num_success / args.num_episodes, end - start))
