import gym
import torch
import torch.optim as optim

class QLearningAgent:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.q_table = torch.zeros((num_states, num_actions))

    def select_action(self, state):
        return self.q_table[state].argmax().item()

    def update_q_value(self, state, action, reward, next_state, alpha, gamma):
        target = reward + gamma * self.q_table[next_state].max()
        self.q_table[state][action] = (1 - alpha) * self.q_table[state][action] + alpha * target

def train_agent(client_id, agent, env, num_episodes=100, alpha=0.1, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(env.spec.max_episode_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update_q_value(state, action, reward, next_state, alpha, gamma)

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Client {client_id}, Episode {episode+1}, Total Reward: {total_reward}")

def main():
    env = gym.make("Taxi-v3")
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    agent1 = QLearningAgent(num_states, num_actions)
    agent2 = QLearningAgent(num_states, num_actions)

    num_episodes = 200

    for episode in range(num_episodes):
        train_agent(1, agent1, env)
        train_agent(2, agent2, env)

        # Federated Aggregation (averaging Q-values)
        agent1.q_table = (agent1.q_table + agent2.q_table) / 2.0

        print(f"Global Model Updated - Episode {episode+1}")

if __name__ == "__main__":
    main()
