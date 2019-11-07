import torch
import torch.nn as nn
import torch.functional as F
import numpy
import gym

class CartPoleModel(nn.Module):
    def __init__(self, action):
        super(CartPoleModel, self).__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 1)
        self.action = action
    
    def forward(self, x):
        x = self.linear1(x).clamp(min=0)
        return self.linear2(x)

model_0 = CartPoleModel(0)
model_1 = CartPoleModel(1)
env = gym.make('CartPole-v1')

optim_0 = torch.optim.SGD(model_0.parameters(), lr=0.007)
optim_1 = torch.optim.SGD(model_1.parameters(), lr=0.007)

criterion = nn.MSELoss(reduction='sum')

for i_episode in range(10000):
    observation = env.reset()
    pred = torch.Tensor()
    optim = optim_0
    for t in range(100):
        env.render()
        o_tensor = torch.Tensor(observation)

        R_pred_0 = model_0(o_tensor)
        R_pred_1 = model_1(o_tensor)
        print('0: ', R_pred_0.item(), ' 1:', R_pred_1.item())

        if R_pred_0 >= R_pred_1:
            optim = optim_0
            pred = R_pred_0
            action = model_0.action
            prev_pred = model_0.action
        else:
            optim = optim_1
            pred = R_pred_1
            action = model_1.action
            prev_pred = model_1.action

        observation, reward, done, info = env.step(action)
        if done: reward = -2
        loss = criterion(pred, torch.Tensor([reward]))
        optim.zero_grad()
        loss.backward()
        optim.step()

        print('ACTION: {}, LOSS: {}, PRED_R: {}, REWARD {}'.format(action, loss.item(), pred.item(), reward))

        if done:
            break
env.close()

