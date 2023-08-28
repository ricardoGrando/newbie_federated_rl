import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.q_network = DQN(input_size, output_size)
        self.target_network = DQN(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.001
        self.output_size = output_size

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_network.output_size - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update_q_network(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        next_q_values = self.target_network(next_states_tensor).max(1)[0].detach()

        target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = nn.functional.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

# Simulated client training function
def train_agent(client_id, agent, env, num_episodes=10, epsilon_decay=0.995):
    epsilon = 1.0

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(env.spec.max_episode_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.add_experience(state, action, reward, next_state, done)
            agent.update_q_network()

            state = next_state

            if done:
                break

        epsilon *= epsilon_decay
        print(f"Client {client_id}, Episode {episode+1}, Total Reward: {total_reward}")

# Main Federated DQN process
def main():
    env = gym.make("CartPole-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    agent1 = DQNAgent(input_size, output_size)
    agent2 = DQNAgent(input_size, output_size)

    num_episodes = 10

    for episode in range(num_episodes):
        train_agent(1, agent1, env)
        train_agent(2, agent2, env)

        # Federated Aggregation (averaging Q-network parameters)
        for target_param, local_param in zip(agent1.target_network.parameters(), agent2.target_network.parameters()):
            print(target_param, local_param)
            target_param.data.copy_((target_param.data + local_param.data) / 2.0)

        print(f"Global Model Updated - Episode {episode+1}")

if __name__ == "__main__":
    main()
