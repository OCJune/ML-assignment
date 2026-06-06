import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from MonteCarlo.mc_agent import MCAgent
from QLearning.q_agent import QLearningAgent
from tqdm import tqdm

def run_mc(episodes=300, hidden_layers=[64, 64]):
    env = Environment()
    agent = MCAgent(hidden_layers=hidden_layers)
    rewards = []
    steps_to_goal = []
    cliff_falls = 0
    
    for e in range(episodes):
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
            
            if step > 200:
                done = True
        
        agent.train_model()
        rewards.append(total_reward)
        steps_to_goal.append(step)
        
    return rewards, steps_to_goal, cliff_falls

def run_qlearning(episodes=300, hidden_layers=[64, 64]):
    env = Environment()
    agent = QLearningAgent(hidden_layers=hidden_layers)
    rewards = []
    steps_to_goal = []
    cliff_falls = 0
    
    for e in range(episodes):
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
                
            if step > 200:
                done = True
                
        rewards.append(total_reward)
        steps_to_goal.append(step)
        
    return rewards, steps_to_goal, cliff_falls

if __name__ == "__main__":
    """
    1. 자동화된 실험: 단일 에이전트 학습에 집중하는 예제코드와 달리, 5가지 각기 다른 신경망 구조에서 
       두 알고리즘을 동시에 테스트하는 자동화 루프를 구현했습니다.
    2. 통계적 비교: 학습 후반부(마지막 50 에피소드)의 평균 보상과 스텝 수를 계산하여 객관적인 성능 지표를 제시합니다.
    3. 시각화 강화: Matplotlib을 사용하여 알고리즘별 수렴 과정을 그래프로 한눈에 비교 분석할 수 있게 했습니다.
    """
    EPISODES = 300
    # 여러 구조 테스트
    architectures = {
        "Shallow-Tiny [32]": [32],
        "Shallow-Mid [32, 32]": [32, 32],
        "Standard [64, 64]": [64, 64],
        "Deep-Slim [32, 32, 32]": [32, 32, 32],
        "Deep-Large [64, 64, 64]": [64, 64, 64]
    }
    
    results_summary = []
    
    num_arch = len(architectures)
    plt.figure(figsize=(15, 6 * num_arch)) # 구조 개수에 따라 세로 길이 자동 조절
    
    for i, (name, layers) in enumerate(architectures.items()):
        print(f"\n--- Testing Architecture: {name} ---")
        
        # MC 실험
        print(f"Running Monte Carlo...")
        mc_rewards, mc_steps, mc_falls = run_mc(EPISODES, hidden_layers=layers)
        
        # Q-Learning 실험
        print(f"Running Q-Learning...")
        q_rewards, q_steps, q_falls = run_qlearning(EPISODES, hidden_layers=layers)
        
        # 결과 요약 저장 (마지막 50 에피소드 평균 보상 등)
        results_summary.append({
            "Arch": name,
            "Algo": "Monte Carlo",
            "AvgReward": np.mean(mc_rewards[-50:]),
            "AvgSteps": np.mean(mc_steps[-50:]),
            "TotalFalls": mc_falls
        })
        results_summary.append({
            "Arch": name,
            "Algo": "Q-Learning",
            "AvgReward": np.mean(q_rewards[-50:]),
            "AvgSteps": np.mean(q_steps[-50:]),
            "TotalFalls": q_falls
        })

        # 그래프 그리기 (num_arch행 2열 격자)
        plt.subplot(num_arch, 2, i*2 + 1)
        plt.plot(mc_rewards, label='MC')
        plt.plot(q_rewards, label='Q-Learning')
        plt.title(f'Reward ({name})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        plt.subplot(num_arch, 2, i*2 + 2)
        plt.plot(mc_rewards, label='MC')
        plt.plot(q_rewards, label='Q-Learning')
        plt.title(f'Steps to Goal ({name})')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('팀플/learning_comparison_multi_arch.png')
    
    # 분석을 위한 표 출력
    print("\n" + "="*80)
    print(" " * 25 + "ALGORITHM PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Architecture':<20} | {'Algorithm':<15} | {'Avg Reward':<12} | {'Avg Steps':<10} | {'Falls':<6}")
    print("-" * 80)
    for res in results_summary:
        print(f"{res['Arch']:<20} | {res['Algo']:<15} | {res['AvgReward']:<12.2f} | {res['AvgSteps']:<10.2f} | {res['TotalFalls']:<6}")
    print("="*80)
    print("\n* Avg Reward/Steps are calculated from the last 50 episodes.")
    print("* Plot saved as '팀플/learning_comparison_multi_arch.png'")
