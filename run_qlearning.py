from environment import Environment
from QLearning.q_agent import QLearningAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    env = Environment()
    agent = QLearningAgent()
    episodes = 500
    rewards = []

    print("Training Q-Learning Agent...")
    for e in tqdm(range(episodes)):
        state = env.start_position
        agent.set_pos(state)
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            state_onehot = agent.state_to_onehot(state)
            action = agent.select_action(state_onehot)
            next_state, reward, done = env.move(agent, action)
            
            agent.train_model(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            if step > 200: # 무한 루프 방지
                done = True
        
        rewards.append(total_reward)

    plt.plot(rewards)
    plt.title("Q-Learning Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

if __name__ == "__main__":
    main()
