import argparse
import gym
import Maze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from visdom import Visdom


class Model(nn.Module):
    def __init__(self, map_size, max_step, num_actions):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(5, 16, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3))
        self.fc1 = torch.nn.Linear((map_size - 4) ** 2 * 16, 256)
        self.embed_a = torch.nn.Embedding(num_actions + 1, 8, padding_idx=0)
        self.embed_t = torch.nn.Embedding(max_step, 8)
        self.policy = torch.nn.Linear(256 + 6*8, num_actions)
        self.value = torch.nn.Linear(256 + 6*8, 1)
        layers = [self.conv1, self.conv2, self.fc1, self.embed_a,
                  self.embed_t, self.policy, self.value]
        for layer in layers:
            torch.nn.init.kaiming_normal(layer.weight)
            if hasattr(layer, 'bias'):
                torch.nn.init.constant(layer.bias, 0)

    def forward(self, belief_map, history, step):
        x = F.relu(self.conv1(belief_map.unsqueeze(0)))
        x = F.relu(self.conv2(x))
        x = x.squeeze(0).view(-1)
        x = F.relu(self.fc1(x))
        a = self.embed_a(history).view(-1)
        t = self.embed_t(step).view(-1)
        x = torch.cat([x, a, t])
        log_pi = F.log_softmax(self.policy(x), dim=-1)
        v = self.value(x)
        return log_pi, v


class A2C:
    # Implementation of N-step Advantage Actor Critic.
    def __init__(self, env, n, use_cuda=False):
        # Initializes A2C.
        # Args:
        # - env: Gym environment.
        # - n: The value of N in N-step A2C.
        self.env = env
        self.n = n
        self.model = Model(env.size, env.max_step, env.action_space.n)
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()

    def _array2var(self, array, requires_grad=True):
        var = Variable(torch.from_numpy(array), requires_grad)
        if self.use_cuda:
            var = var.cuda()
        return var

    def train(self, args):
        policy_losses = np.zeros(args.train_episodes)
        value_losses = np.zeros(args.train_episodes)
        success_rate = np.zeros(args.train_episodes // args.episodes_per_eval + 1)

        vis = Visdom()
        success_rate[0] = self.eval(args.test_episodes)
        print('episode %d\t success rate %.2f' % (0, success_rate[0]))
        opts = dict(xlabel='episodes', ylabel='success rate')
        reward_plot = vis.line(X=np.array([0]), Y=success_rate[:1], env=args.task_name, opts=opts)
        policy_loss_plot = None
        value_loss_plot = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr)
        for i in range(args.train_episodes):
            if i % args.episodes_per_mapgen:
                self.env.seed(np.random.randint(1000, 2**31))
                self.env.generate_map()
            policy_losses[i], value_losses[i] = self.train_one_episode(args.gamma)
            if (i + 1) % args.episodes_per_plot == 0:
                if policy_loss_plot is None:
                    opts = dict(xlabel='episodes', ylabel='policy loss')
                    policy_loss_plot = vis.line(X=np.arange(1, i + 2), Y=policy_losses[:i + 1],
                                                env=args.task_name, opts=opts)
                else:
                    vis.line(X=np.arange(i - args.episodes_per_plot + 1, i + 2),
                             Y=policy_losses[i - args.episodes_per_plot:i + 1],
                             env=args.task_name, win=policy_loss_plot, update='append')
                if value_loss_plot is None:
                    opts = dict(xlabel='episodes', ylabel='value loss')
                    value_loss_plot = vis.line(X=np.arange(1, i + 2), Y=value_losses[:i + 1],
                                               env=args.task_name, opts=opts)
                else:
                    vis.line(X=np.arange(i - args.episodes_per_plot + 1, i + 2),
                             Y=value_losses[i - args.episodes_per_plot:i + 1],
                             env=args.task_name, win=value_loss_plot, update='append')
            if (i + 1) % args.episodes_per_eval == 0:
                j = (i + 1) // args.episodes_per_eval
                success_rate[j] = self.eval(args.test_episodes)
                print('episode %d\t policy loss %.6f\t value loss %.6f\t success rate %.2f' % (
                    i + 1, policy_losses[i], value_losses[i], success_rate[j]))
                vis.line(X=np.array([i+1]), Y=success_rate[j:j+1],
                         env=args.task_name, win=reward_plot, update='append')
        torch.save(a2c.model.state_dict(), 'models/' + args.task_name + '.model')

    def train_one_episode(self, gamma):
        # Trains the model on a single episode using A2C.
        rewards, log_pi, value, entropy = self.generate_episode()
        T = len(rewards)
        R = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            v_end = value.data[t + self.n] if t + self.n < T else 0
            R[t] = gamma ** self.n * v_end + \
                   sum([gamma ** k * rewards[t+k] for k in range(min(self.n, T - t))])
        R = self._array2var(R, requires_grad=False)
        policy_loss = -(log_pi * (R - value.detach()) + args.beta * entropy).mean()
        value_loss = ((R - value) ** 2).mean()
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss.data[0], value_loss.data[0]

    def eval(self, num_episodes, stochastic=True):
        # Tests the model on n episodes
        total_success = 0
        for i in range(num_episodes):
            self.env.seed(i)
            self.env.generate_map()
            rewards, _, _, _ = self.generate_episode(render=False, stochastic=stochastic)
            total_success += rewards[-1]
        #print(total_success, num_episodes, total_success/num_episodes)
        return total_success*1.0 / num_episodes

    def select_action(self, belief, map, history, step, stochastic):
        # Select the action to take by sampling from the policy model
        # Returns
        # - the action
        # - log probability of the chosen action (as a Variable)
        # - value of the state (as a Variable)
        belief_map = np.concatenate([belief, np.expand_dims(map, 0)], axis=0)
        log_pi, value = self.model(self._array2var(belief_map),
                                   self._array2var(history, requires_grad=False),
                                   self._array2var(np.array([step]), requires_grad=False))
        entropy = -(log_pi.exp() * log_pi).sum()
        if stochastic:
            action = torch.distributions.Categorical(log_pi.exp()).sample()
        else:
            _, action = log_pi.max(0)
        return action.data[0], log_pi[action], value, entropy

    def generate_episode(self, render=False, stochastic=True):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of rewards, indexed by time step
        # - a Variable of log probabilities
        # - a Variable of state values
        log_probs = []
        values = []
        entropies = []
        rewards = []
        history = np.zeros(5, dtype=np.int64)
        belief = self.env.reset()
        step = 0
        done = False
        while not done:
            if render:
                self.env.render()
            action, log_prob, value, entropy = self.select_action(
                belief, self.env.map, history, step, stochastic)
            belief, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            history = np.roll(history, 1)
            history[0] = action + 1
            step += 1
        return rewards, torch.cat(log_probs), torch.cat(values), torch.cat(entropies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', dest='task_name',
                        default='A2C', help="Name of the experiment")
    parser.add_argument('--env_name', dest='env_name',
                        default='Maze-v1', help="Name of the environment.")
    parser.add_argument('--size', dest='size', type=int,
                        default=15, help="Size of the random maze.")
    parser.add_argument('--max_step', dest='max_step', type=int,
                        default=20, help="Maximum number of steps in an episode.")
    parser.add_argument('-n', dest='n', type=int,
                        default=20, help="Number steps in the trace.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.001, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="The discount factor.")
    parser.add_argument('--beta', dest='beta', type=float,
                        default=0.01, help="The entropy weight.")
    parser.add_argument('--train_episodes', dest='train_episodes', type=int,
                        default=200000, help="Number of episodes to train on.")
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help="Number of episodes to test on.")
    parser.add_argument('--episodes_per_eval', dest='episodes_per_eval', type=int,
                        default=1000, help="Number of episodes between each evaluation.")
    parser.add_argument('--episodes_per_plot', dest='episodes_per_plot', type=int,
                        default=50, help="Number of episodes between each plot update.")
    parser.add_argument('--episodes_per_mapgen', dest='episodes_per_mapgen', type=int,
                        default=100, help="Number of episodes between each random map generation.")
    parser.add_argument('--use_cuda', action='store_true', help='Use GPU in training.')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.set_size(args.size)
    env.set_max_step(args.max_step)

    a2c = A2C(env, args.n, args.use_cuda)
    a2c.train(args)
