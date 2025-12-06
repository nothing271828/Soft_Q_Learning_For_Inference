import numpy as np
import torch
import torch.nn.functional as func
import copy

from Environment import DyckNTaskEnvironment
from QNetwork import QNetwork


class Agent(object):
    def __init__(self, environment: DyckNTaskEnvironment, q_network: QNetwork, input_vocabulary_size,
                 output_vocabulary_size, sentence_length, buffer_size, device=None):
        """
        :param environment:
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.environment = environment
        self.network = q_network.to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.target_network.eval()  # Target网络不需要计算梯度

        self.replay_buffer = ReplayBuffer((input_vocabulary_size, sentence_length), output_vocabulary_size,
                                          max_size=buffer_size, device=self.device)

        self.sentence_length = sentence_length
        self.input_vocabulary_size = input_vocabulary_size
        self.output_vocabulary_size = output_vocabulary_size

        self.gamma = 0.95
        self.learning_rate = 0.001
        self.tau = 0.005

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.counter = 0

    def soft_update(self):
        """
        theta_target = tau * theta_current + (1 - tau) * theta_target
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(target_param.data * self.tau + param.data * (1.0 - self.tau))

    def train_step(self, beta, batch_size):
        """
        单步训练函数，
        :param beta:
        :param batch_size:
        :return:
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)  # 按顺序分别为state、action、reward、next state、done

        self.optimizer.zero_grad()
        Q_values = self.network(state, action)  # shape [B, num_actions]

        y = reward + self.gamma * self.network.get_value(next_state, beta)

        # if self.counter % 100 == 0:
        #     print(f"target: {y.mean():.4f}, Q: {Q_values.mean():.4f}")
        # self.counter += 1

        loss = func.mse_loss(Q_values, y)
        loss.backward()

        self.optimizer.step()
        self.soft_update()

        return loss.item()

    def train(self, num_episodes, batch_size, beta, question_length):
        """
        完整训练循环
        """
        print(f"Start Training: Episodes={num_episodes}, Beta={beta}")

        # 预热：先随机跑一点数据填充 Buffer，防止一开始训练时相关性太强
        # (可选，但推荐)
        if self.replay_buffer.size < batch_size * 5:
            print("Warming up replay buffer...")
            self.generate_buffer_steps(n_steps=batch_size * 10, question_length=question_length, beta=beta)

        total_steps = 0

        for episode in range(num_episodes):
            # 初始化环境状态
            state = self.environment.generate_initial_state(question_length, self.sentence_length)
            episode_reward = 0
            episode_loss = 0
            steps_in_episode = 0
            done = False

            while not done:
                action = self.network.get_action(state, beta)
                # 执行动作
                next_state, reward, done = self.environment.step(state, action)

                # 存入 Buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                # 更新状态
                state = next_state

                # 每走一步，训练一步 (或者每隔几步训练一次)
                loss = self.train_step(beta, batch_size)

                # 记录数据
                episode_reward += reward
                episode_loss += loss
                steps_in_episode += 1
                total_steps += 1

            # 打印日志
            avg_loss = episode_loss / steps_in_episode
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {episode_reward:.4f} | "
                      f"Avg Loss: {avg_loss:.6f} | "
                      f"Steps: {steps_in_episode}")

    def generate_buffer(self, question_length, beta):
        while self.replay_buffer.size < self.replay_buffer.max_size:
            done = False
            state = self.environment.generate_initial_state(question_length, self.sentence_length)
            while not done:
                action = self.network.get_action(state, beta)
                next_state, reward, done = self.environment.step(state, action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

    def generate_buffer_steps(self, n_steps, question_length, beta):
        """
        辅助函数：仅用于预热 Buffer，不进行训练
        """
        steps = 0
        while steps < n_steps:
            state = self.environment.generate_initial_state(question_length, self.sentence_length)
            done = False
            while not done and steps < n_steps:
                action = self.network.get_action(state, beta)
                next_state, reward, done = self.environment.step(state, action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                steps += 1


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100, device=None):
        """
        :param state_dim  (int or tuple)状态空间的维度，例如 8 或 (3, 84, 84)
        :param action_dim (int): 动作空间的维度
        :param max_size (int): Buffer的最大容量
        :param device (torch.device): 训练使用的设备 (CPU/CUDA)
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sd = (max_size,) + state_dim
        self.state_buffer = torch.zeros(sd, dtype=torch.float32, device=self.device)
        self.next_state_buffer = torch.zeros(sd, dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.zeros(max_size, dtype=torch.float32, device=self.device)
        self.done_buffer = torch.zeros(max_size, dtype=torch.int16, device=self.device)

    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验数据
        """
        idx = self.ptr % self.max_size

        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.next_state_buffer[idx] = next_state
        # 将 boolean done 转换为 float (0.0 或 1.0)，方便后续计算 Bellman error
        self.done_buffer[idx] = 1.0 if done else 0.0

        self.ptr = self.ptr + 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        随机采样一个 batch 并转换为 Tensor
        """
        # 生成随机索引
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state_buffer[ind],
            self.action_buffer[ind],
            self.reward_buffer[ind],
            self.next_state_buffer[ind],
            self.done_buffer[ind]
        )

    def __len__(self):
        return self.size
