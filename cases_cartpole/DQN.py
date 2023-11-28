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


# modified from https://github.com/boyu-ai/Hands-on-RL
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity, device):
        # 用队列存储数据，避免溢出
        self.buffer = collections.deque(maxlen=capacity) 
        self.device = device 

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    # 从buffer中采样数据,数量为batch_size
    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.tensor(action).view(-1, 1).to(self.device)
        reward = torch.FloatTensor(reward).view(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        return state, action, reward, next_state, done

    # 目前buffer中数据的数量
    def size(self):  
        return len(self.buffer)
    
class Net(nn.Module):
    ''' 定义神经网络结构 '''
    def __init__(self, state_space, hidden_space, action_space):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space)
        )
        
    def forward(self, x):
        action_value = self.fc(x)
        return action_value
    
class DQN:
    ''' 定义DQN算法 '''
    def __init__(self, state_space, hidden_space, action_space, learning_rate, gamma, epsilon, update_frequency, device):
        self.action_space = action_space
        self.q_eval = Net(state_space, hidden_space, action_space).to(device) 
        self.q_target = Net(state_space, hidden_space, action_space).to(device)   #目标网络
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_frequency = update_frequency
        self.count = 0
        self.device = device
        self.writer = SummaryWriter('./dqn_logs')
        
    def choose_action(self, state):     
        '''  epsilon-greedy策略采取动作 '''
        #选择随机动作进行探索
        if np.random.uniform() < self.epsilon:      
            action = np.random.randint(0, self.action_space)
        #选择最优动作    
        else:       
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_eval(state).argmax().item()
        return action
    
    def update(self):
        ''' 更新Q网络 '''
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Q值
        q_values = self.q_eval(states).gather(1, actions)  
        # 下个状态的最大Q值
        max_next_q_values = self.q_target(next_states).max(1)[0].view(-1, 1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  
        # 均方误差损失函数
        loss = self.loss_func(q_values, q_targets)
        # 梯度清零
        self.optimizer.zero_grad()  
        # 反向传播梯度
        loss.backward() 
        # 更新参数
        self.optimizer.step()
        
        self.writer.add_scalar('Loss/train', loss, self.count)

        # 更新目标网络
        if self.count % self.update_frequency == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())  
        self.count += 1

def moving_average(a, window_size):
    ''' 平滑reward曲线 '''
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 定义超参数
lr = 1e-3
num_episodes = 1000
iter = 10
gamma = 0.99
epsilon = 0.01
target_update = 10
buffer_size = 10000
batch_size = 64
minimal_size = 500
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = "CartPole-v1"
env = gym.make(env_name)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
hidden_space = 64
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = DQN(state_space, hidden_space, action_space, lr, gamma, epsilon, target_update, device)
replay_buffer = ReplayBuffer(buffer_size, device)
reward_list = []

# 训练过程
for i in range(iter):
    with tqdm(total=int(num_episodes / iter), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / iter)):
            episode_raward = 0
            state, info = wrapped_env.reset(seed=0)
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = wrapped_env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_raward += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    agent.update()
            reward_list.append(episode_raward)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(reward_list[-10:])
                })
            pbar.update(1)

# 绘制结果
episodes_list = list(range(len(reward_list)))
plt.figure(1)
plt.plot(episodes_list, reward_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('./fig/dqn_return1.png')

mv_return = moving_average(reward_list, 19)
plt.figure(2)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('./fig/dqn_return2.png')
plt.show()    
            
        
        