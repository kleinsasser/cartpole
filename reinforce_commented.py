import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# handy feature allowing the modification of some hyperparameters through command line arguments
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128) # fully connected layer 1
        self.dropout = nn.Dropout(p=0.6) # dropout layer
        self.affine2 = nn.Linear(128, 2) # fully connected layer 2

        self.saved_log_probs = [] # added capability of storing previous actions
        self.rewards = [] # rewards earned by particular actions

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1) # maps action scores to probabilities [0,1]

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item() # defining a epsilon as the smallest value representable by a 32 bit float

# takes state from gym as argument, returns next action as calculated by the policy
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0) # reshape state data
    probs = policy(state) # calculate the probability of each action from the policy
    m = Categorical(probs) # transforms probs to categorical probabilities between 0 and 1
    action = m.sample() # takes random action from categorical probabilities
    policy.saved_log_probs.append(m.log_prob(action)) # store chosen action as addable log probability
    return action.item()


def finish_episode():
    R = 0 # running reward is 0
    policy_loss = [] # init some storage arrays w/ descriptive names
    returns = []
    for r in policy.rewards[::-1]: # iterate through rewards in reverse order (most to least recent)
        R = r + args.gamma * R # sum up rewards (haven't figured out gamma yet)
        returns.insert(0, R) # store each R for each r
    returns = torch.tensor(returns)
    # transform return values to their corresponding value on a normal curve. eps preventing division by 0
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # at this point the returns tensor contains the rewards as above or below average which map to actions taken in saved_log_probs
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R) # calculate loss for each timestep
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum() # add up total loss
    policy_loss.backward() # calculate gradient
    optimizer.step() # optimize
    del policy.rewards[:] # clear rewards and log_probs
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1): # poorly named infite iterator with step size 1
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action) # gather new state data and reward from previous action
            if args.render: # optionally render the scene
                env.render()
            policy.rewards.append(reward) # record reward for timestep
            ep_reward += reward # store accumulated rewards
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward # seemingly arbitrary running reward formula
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold: # end if reward achieves threshold
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

