import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from MonteCarlo.mc_agent import MCAgent
from QLearning.q_agent import QLearningAgent
from tqdm import tqdm

def run_mc(episodes=500):
    env = Environment()
    agent = MCAgent()
    rewards = []
    steps_to_goal = []
    cliff_falls = 0
    
    print("Starting Monte Carlo training...")
    for e in tqdm(range(episodes)):
        state = env.start_position
        agent.set_pos(state)
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            state_onehot = agent.state_to_onehot(state)
            action = agent.select_action(state_onehot)
            next_state, reward, done = env.move(agent, action)
            
            agent.append_sample(state, action, reward)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if reward == env.cliff:
                cliff_falls += 1
            
            if step > 200: # 제한된 스텝 수
                done = True
        
        agent.train_model()
        rewards.append(total_reward)
        steps_to_goal.append(step)
        
    return rewards, steps_to_goal, cliff_falls

def run_qlearning(episodes=500):
    env = Environment()
    agent = QLearningAgent()
    rewards = []
    steps_to_goal = []
    cliff_falls = 0
    
    print("Starting Q-Learning training...")
    for e in tqdm(range(episodes)):
        state = env.start_position
        agent.set_pos(state)
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            state_onehot = agent.state_to_onehot(state)
            action = agent.select_action(state_onehot)
            next_state, reward, done = env.move(agent, action)
            
            agent.train_model(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if reward == env.cliff:
                cliff_falls += 1
                
            if step > 200: # 제한된 스텝 수
                done = True
                
        rewards.append(total_reward)
        steps_to_goal.append(step)
        
    return rewards, steps_to_goal, cliff_falls

if __name__ == "__main__":
    EPISODES = 300 # 비교를 위한 에피소드 수
    
    mc_rewards, mc_steps, mc_falls = run_mc(EPISODES)
    q_rewards, q_steps, q_falls = run_qlearning(EPISODES)
    
    # 결과 시각화
    plt.figure(figsize=(12, 10))
    
    # 1. 에피소드별 총 보상 비교
    plt.subplot(2, 1, 1)
    plt.plot(mc_rewards, label='Monte Carlo')
    plt.plot(q_rewards, label='Q-Learning')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # 2. 에피소드별 스텝 수 비교
    plt.subplot(2, 1, 2)
    plt.plot(mc_steps, label='Monte Carlo')
    plt.plot(q_steps, label='Q-Learning')
    plt.title('Steps to Goal per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('팀플/learning_comparison.png')
    plt.show()
    
    print(f"\nTraining Results (Total {EPISODES} episodes):")
    print(f"Monte Carlo - Total Cliff Falls: {mc_falls}")
    print(f"Q-Learning  - Total Cliff Falls: {q_falls}")
    print("Comparison plot saved as '팀플/learning_comparison.png'")
