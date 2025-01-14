import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PolicyNet(torch.nn.Module):
    ''' 定义神经网络结构 '''
    def __init__(self, state_space, hidden_space, action_space):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        action_prob = self.fc(x)
        return action_prob
    
class REINFORCE:
    ''' 定义REINFORCE算法 '''
    def __init__(self, state_space, hidden_space, action_space, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_space, hidden_space, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) 
        self.gamma = gamma 
        self.device = device

    def take_action(self, state):  
        ''' 根据动作概率分布随机采样 '''
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        ''' 更新网络 '''
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        # 从最后一步算起
        for i in reversed(range(len(reward_list))):  
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            # 每一步的损失函数
            loss = -log_prob * G  
            # 反向传播
            loss.backward() 
        # 更新参数    
        self.optimizer.step() 
        
def moving_average(a, window_size):
    ''' 平滑reward曲线 '''
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 定义超参数
learning_rate = 1e-3
num_episodes = 1000
iter = 10
gamma = 0.99
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = "CartPole-v1"
env = gym.make(env_name)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
torch.manual_seed(0)
hidden_space = 128
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = REINFORCE(state_space, hidden_space, action_space, learning_rate, gamma, device)
return_list = []

# 训练过程
for i in range(iter):
    with tqdm(total=int(num_episodes / iter), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / iter)):
            episode_return = 0
            state, info = wrapped_env.reset(seed=0)
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = wrapped_env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# 绘制结果         
episodes_list = list(range(len(return_list)))
plt.figure(1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('./fig/pg_return1.png')

mv_return = moving_average(return_list, 9)
plt.figure(2)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('./fig/pg_return2.png')
plt.show()            