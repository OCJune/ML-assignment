import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from environment import Environment
from MonteCarlo.mc_agent import MCAgent
from QLearning.q_agent import QLearningAgent

# MAX STEP을 제한하여 학습의 길이를 줄인다. 
MAX_STEPS = 200
# 에피소드 20000개
EPISODES = 20000
EVAL_EPISODES = 50


def get_exploring_start_states(env):
    """2장 Monte Carlo exploring starts를 위한 road 상태 목록"""
    states = []
    for row in range(env.reward.shape[0]):
        for col in range(env.reward.shape[1]):
            if env.reward_list1[row][col] == "road":
                states.append(np.array([row, col]))
    return states


def run_episode(env, agent, algo, train=True, max_steps=MAX_STEPS, start_state=None, first_action=None):
    state = np.array(env.start_position if start_state is None else start_state)
    agent.set_pos(state)

    total_reward = 0
    steps = 0
    falls = 0
    done = False

    while not done and steps < max_steps:
        if steps == 0 and first_action is not None:
            action = first_action
        else:
            action = agent.select_action(state)

        next_state, reward, done = env.move(agent, action)
        next_state = np.array(next_state)

        if train:
            if algo == "MC":
                agent.append_sample(state, action, reward)
            elif algo == "QL":
                agent.train_model(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1
        falls += int(reward == env.cliff)

    return total_reward, steps, falls, done


def evaluate_agent(agent, algo, episodes=EVAL_EPISODES):
    env = Environment()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    success_steps = []
    falls = 0
    successes = 0

    for _ in range(episodes):
        reward, steps, episode_falls, success = run_episode(env, agent, algo, train=False)
        rewards.append(reward)
        falls += episode_falls

        if success:
            successes += 1
            success_steps.append(steps)

    agent.epsilon = old_epsilon

    return {
        "EvalReward": np.mean(rewards),
        "EvalSteps": np.mean(success_steps) if success_steps else np.nan,
        "EvalFalls": falls,
        "EvalSuccessRate": successes / episodes,
    }


def get_greedy_path(agent, max_steps=MAX_STEPS):
    """학습된 greedy 정책으로 시작점부터 이동한 경로를 기록"""
    env = Environment()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = np.array(env.start_position)
    agent.set_pos(state)

    path = [tuple(state)]
    rewards = []
    actions = []
    done = False

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.move(agent, action)
        next_state = np.array(next_state)

        actions.append(action)
        rewards.append(reward)
        path.append(tuple(next_state))

        state = next_state
        if done:
            break

    agent.epsilon = old_epsilon
    return path, actions, rewards, done


def train_agent(algo, episodes=EPISODES, seed=0):
    np.random.seed(seed)
    env = Environment()
    agent = MCAgent(env) if algo == "MC" else QLearningAgent()
    exploring_states = get_exploring_start_states(env)

    rewards = []
    steps = []
    falls = []
    successes = []

    for _ in tqdm(range(episodes), desc=algo):
        if algo == "MC":
            start_state = exploring_states[np.random.randint(len(exploring_states))]
            first_action = np.random.randint(agent.action_size)
        else:
            start_state = None
            first_action = None

        reward, episode_steps, episode_falls, success = run_episode(
            env,
            agent,
            algo,
            train=True,
            start_state=start_state,
            first_action=first_action,
        )

        if algo == "MC":
            agent.train_model()
        else:
            agent.decay_epsilon()

        rewards.append(reward)
        steps.append(episode_steps)
        falls.append(episode_falls)
        successes.append(int(success))

    return {
        "agent": agent,
        "rewards": np.array(rewards),
        "steps": np.array(steps),
        "falls": np.array(falls),
        "successes": np.array(successes),
        "eval": evaluate_agent(agent, algo),
    }


def summarize_result(algo, result, last_n=50):
    return {
        "Algo": algo,
        "TrainRewardLastN": np.mean(result["rewards"][-last_n:]),
        "TrainStepsLastN": np.mean(result["steps"][-last_n:]),
        "TrainFalls": int(np.sum(result["falls"])),
        "TrainSuccessRate": np.mean(result["successes"]),
        **result["eval"],
    }


def print_summary(rows):
    print("\n" + "=" * 118)
    print(" " * 42 + "MC vs Q-Learning Comparison")
    print("=" * 118)
    print(
        f"{'Algo':<12} | {'Train Reward':<13} | {'Train Steps':<12} | {'Train Falls':<11} | "
        f"{'Train Success':<13} | {'Eval Reward':<11} | {'Eval Steps':<10} | {'Eval Success':<12}"
    )
    print("-" * 118)

    for row in rows:
        eval_steps = "N/A" if np.isnan(row["EvalSteps"]) else f"{row['EvalSteps']:.2f}"
        print(
            f"{row['Algo']:<12} | {row['TrainRewardLastN']:<13.2f} | {row['TrainStepsLastN']:<12.2f} | "
            f"{row['TrainFalls']:<11} | {row['TrainSuccessRate'] * 100:<12.1f}% | "
            f"{row['EvalReward']:<11.2f} | {eval_steps:<10} | {row['EvalSuccessRate'] * 100:<11.1f}%"
        )

    print("=" * 118)
    print("* Train Reward/Steps are averages from the last 50 training episodes.")
    print("* Eval metrics are measured after training with epsilon=0 greedy policy.")


def plot_training(mc_result, ql_result):
    os.makedirs("팀플", exist_ok=True)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(mc_result["rewards"], label="Monte Carlo")
    plt.plot(ql_result["rewards"], label="Q-Learning")
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(mc_result["steps"], label="Monte Carlo")
    plt.plot(ql_result["steps"], label="Q-Learning")
    plt.title("Training Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(np.cumsum(mc_result["successes"]) / (np.arange(len(mc_result["successes"])) + 1), label="Monte Carlo")
    plt.plot(np.cumsum(ql_result["successes"]) / (np.arange(len(ql_result["successes"])) + 1), label="Q-Learning")
    plt.title("Cumulative Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(np.cumsum(mc_result["falls"]), label="Monte Carlo")
    plt.plot(np.cumsum(ql_result["falls"]), label="Q-Learning")
    plt.title("Cumulative Cliff Falls")
    plt.xlabel("Episode")
    plt.ylabel("Falls")
    plt.legend()

    plt.tight_layout()
    plt.savefig("팀플/learning_comparison_mc_vs_ql.png")


def plot_policy_path(agent, title, filename):
    """학습된 greedy 정책의 실제 이동 경로를 격자 그림으로 저장"""
    os.makedirs("팀플", exist_ok=True)

    env = Environment()
    path, actions, rewards, success = get_greedy_path(agent)

    rows, cols = env.reward.shape
    color_grid = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            if env.reward_list1[row][col] == "cliff":
                color_grid[row, col] = -1
            elif env.reward_list1[row][col] == "goal":
                color_grid[row, col] = 2

    for row, col in path:
        if env.reward_list1[row][col] == "road":
            color_grid[row, col] = 1

    start = tuple(env.start_position)
    goal = tuple(env.goal_position)
    color_grid[start] = 3
    color_grid[goal] = 2

    cmap = plt.matplotlib.colors.ListedColormap([
        "#ef4444",  # cliff
        "#f8fafc",  # road
        "#60a5fa",  # path
        "#22c55e",  # goal
        "#facc15",  # start
    ])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(14, 5))
    plt.imshow(color_grid, cmap=cmap, norm=norm)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.grid(which="major", color="#334155", linewidth=1)
    plt.tick_params(bottom=False, left=False)

    for idx, (row, col) in enumerate(path):
        label = "S" if (row, col) == start else "G" if (row, col) == goal else str(idx)
        plt.text(col, row, label, ha="center", va="center", color="#0f172a", fontsize=10, fontweight="bold")

    action_symbol = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    for (row, col), action in zip(path[:-1], actions):
        plt.text(col + 0.28, row - 0.28, action_symbol[action], ha="center", va="center", color="#111827", fontsize=12)

    total_reward = sum(rewards)
    plt.title(f"{title} Greedy Path | Success: {success} | Steps: {len(actions)} | Reward: {total_reward}")
    plt.tight_layout()
    plt.savefig(filename)
    return path, actions, rewards, success


def plot_all_policy_paths(mc_result, ql_result):
    mc_path = plot_policy_path(
        mc_result["agent"],
        "Monte Carlo",
        "팀플/monte_carlo_greedy_path.png",
    )
    ql_path = plot_policy_path(
        ql_result["agent"],
        "Q-Learning",
        "팀플/q_learning_greedy_path.png",
    )
    return mc_path, ql_path


if __name__ == "__main__":
    """
    2장 코드 기반:
    - Monte Carlo control: episode return G를 Q-table 방문 평균으로 업데이트
    - Monte Carlo는 2장 방식처럼 exploring starts로 다양한 state-action을 방문
    - Q-learning: TD(0) target으로 Q-table을 매 스텝 업데이트

    4장 코드 기반 비교 요소:
    - Agent/player 클래스 구조
    - epsilon-greedy policy
    - 학습 루프에서 reward, step, success, fall을 기록하고 두 알고리즘을 비교
    """
    mc_result = train_agent("MC", episodes=EPISODES, seed=0)
    ql_result = train_agent("QL", episodes=EPISODES, seed=0)

    rows = [
        summarize_result("Monte Carlo", mc_result),
        summarize_result("Q-Learning", ql_result),
    ]

    print_summary(rows)
    plot_training(mc_result, ql_result)
    plot_all_policy_paths(mc_result, ql_result)
    print("\n* Plot saved as '팀플/learning_comparison_mc_vs_ql.png'.")
    print("* Greedy paths saved as '팀플/monte_carlo_greedy_path.png' and '팀플/q_learning_greedy_path.png'.")
