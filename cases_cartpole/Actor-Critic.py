import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PolicyNet(torch.nn.Module):
    ''' 策略网络 '''
    def __init__(self, state_space, hidden_space, action_space):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, action_space),
            nn.Softmax(dim=1) )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class ValueNet(torch.nn.Module):
    ''' 价值网络 '''
    def __init__(self, state_space, hidden_space):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1)
        )

    def forward(self, x):
        action_value = self.fc(x)
        return action_value
    
class ActorCritic:
    def __init__(self, state_space, hidden_space, action_space, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_space, hidden_space, action_space).to(device)
        # 价值网络
        self.critic = ValueNet(state_space, hidden_space).to(device)  
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) 
        self.loss_func = nn.MSELoss() 
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

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

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # 时序差分误差
        td_delta = td_target - self.critic(states)  
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = self.loss_func(self.critic(states), td_target.detach())
        # 梯度清零
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 计算梯度
        actor_loss.backward()  
        critic_loss.backward()
        # 更新参数
        self.actor_optimizer.step()  
        self.critic_optimizer.step()
        
def moving_average(a, window_size):
    ''' 平滑reward曲线 '''
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 定义超参数        
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
iter = 10
hidden_space = 128
gamma = 0.97
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = 'CartPole-v1'
env = gym.make(env_name)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
torch.manual_seed(0)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = ActorCritic(state_space, hidden_space, action_space, actor_lr, critic_lr, gamma, device)
return_list = []

# 训练过程
for i in range(iter):
    with tqdm(total=int(num_episodes/iter), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/iter)):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state, info = wrapped_env.reset(seed=0)
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
            if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

# 绘制结果
episodes_list = list(range(len(return_list)))
plt.figure(1)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.savefig('./fig/AC_return1.png')

mv_return = moving_average(return_list, 9)
plt.figure(2)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.savefig('./fig/AC_return2.png')
plt.show()

   