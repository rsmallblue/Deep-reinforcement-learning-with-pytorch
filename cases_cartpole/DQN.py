import random
import numpy as np
import collections
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value
    
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, update_frequency, device):
        self.action_dim = action_dim
        self.q_eval = Net(state_dim, hidden_dim, action_dim).to(device) 
        self.q_target = Net(state_dim, hidden_dim, action_dim).to(device)   #目标网络
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_frequency = update_frequency
        self.count = 0
        self.device = device
        self.writer = SummaryWriter('./case_cartpole/dqn_logs')
        
    def choose_action(self, state):     #epsilon-贪婪策略采取动作
        if np.random.uniform() < self.epsilon:      #选择随机动作进行探索
            action = np.random.randint(0, self.action_dim)
        else:       #选择最优动作
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_eval(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_eval(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.q_target(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.update_frequency == 0:
            self.q_target.load_state_dict(
                self.q_eval.state_dict())  # 更新目标网络
        self.count += 1

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
   
lr = 1e-3
num_episodes = 1000
iter = 10
gamma = 0.99
epsilon = 0.01
target_update = 10
buffer_size = 10000
batch_size = 64
minimal_size = 500
device = torch.device("cude") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
hidden_dim = 64
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
replay_buffer = ReplayBuffer(buffer_size)
return_list = []


for i in range(iter):
    with tqdm(total=int(num_episodes / iter), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / iter)):
            episode_return = 0
            state, info = wrapped_env.reset(seed=0)
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = wrapped_env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            agent.writer.add_scalar('live/finish_step', i_episode+1, global_step=iter)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.figure(1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('./fig/dqn_return1.png')

mv_return = moving_average(return_list, 9)
plt.figure(2)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('./fig/dqn_return2.png')
plt.show()    
            
        
        