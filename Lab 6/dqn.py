"""DLP DQN Lab"""
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'

import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        """
        Append (state, action, reward, next_state, done) to the buffer
        :param transition: (state, action, reward, next_state, done)
        :return: None
        """
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        """
        Sample a batch of transition tensors
        :param batch_size: batch size
        :param device: training device
        :return: a batch of transition tensors
        """
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=state_dim,
                      out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim,
                      out_features=action_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)

        # Initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())

        # TODO
        # self._optimizer = ?
        self._optimizer = Adam(self._behavior_net.parameters(), lr=args.lr)

        # Memory
        self._memory = ReplayMemory(capacity=args.capacity)

        # Config
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        """
        epsilon-greedy based on behavior network
        :param state: current state
        :param epsilon: probability
        :param action_space: action space of current game
        :return: an action
        """
        # TODO
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self._behavior_net.eval()
            with torch.no_grad():
                action_values = self._behavior_net(state)
            self._behavior_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(action_space.n))

    def append(self, state, action, reward, next_state, done):
        """
        Append a step to the memory
        :param state: current state
        :param action: best action
        :param reward: reward
        :param next_state: next state
        :param done: whether the game is finished
        :return: None
        """
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        """
        Update behavior networks and target networks
        :return: None
        """
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            # TODO DQN
            # self._update_target_network()
            # TODO DDQN
            self._soft_update_target_network()

    def _update_behavior_network(self, gamma):
        """
        Update behavior network
        :param gamma: gamma
        :return: None
        """
        # Sample a mini-batch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        # TODO DQN
        # q_value = ?
        # with torch.no_grad():
        #    q_next = ?
        #    q_target = ?
        # criterion = ?
        # loss = criterion(q_value, q_target)
        # q_value = self._behavior_net(state).gather(1, action.long())
        # with torch.no_grad():
        #     q_next = self._target_net(next_state).detach().max(1)[0].unsqueeze(1)
        #     q_target = reward + (gamma * q_next * (1 - done))

        # TODO DDQN
        q_value = self._behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_argmax = self._behavior_net(next_state).detach().max(1)[1].unsqueeze(1)
            q_next = self._target_net(next_state).detach().gather(1, q_argmax)
            q_target = reward + (gamma * q_next * (1 - done))

        loss = nn.MSELoss()(q_value, q_target)

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        """
        Update target network by copying from behavior network
        :return: None
        """
        # TODO
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def _soft_update_target_network(self, tau=.9):
        """
        Update target network by _soft_ copying from behavior network
        :param tau: weight
        :return: None
        """
        for target, behavior in zip(self._target_net.parameters(), self._behavior_net.parameters()):
            target.data.copy_(tau * behavior.data + (1.0 - tau) * target.data)

    def save(self, model_path, checkpoint=False):
        """
        Save behavior networks (and target networks and optimizers) into model_path
        :param model_path: name of the stored model
        :param checkpoint: whether to store target networks and optimizers
        :return: None
        """
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        """
        Load behavior networks (and target networks and optimizers) from model_path
        :param model_path: name of the stored model
        :param checkpoint: whether target networks and optimizers are stored in the model path
        :return: None
        """
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    """
    Training
    :param args: arguments
    :param env: environment
    :param agent: agent
    :param writer: Tensorboard writer
    :return: None
    """
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # Select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # Execute action
            next_state, reward, done, _ = env.step(action)
            # Store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward,
                                epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    """
    Testing
    :param args: arguments
    :param env: environment
    :param agent: agent
    :param writer: Tensorboard writer
    :return: None
    """
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        # TODO
        # ...
        #     if done:
        #         writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
        #         ...
        for _ in range(1000):
            action = agent.select_action(state, 0, action_space)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # Training arguments
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # Testing arguments
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    # Main
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
