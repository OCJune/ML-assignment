# 몬테카를로 vs Q-Learning 성능 비교 분석 프로젝트

이 프로젝트는 `environment.py`에 정의된 **Cliff Walking(절벽 걷기)** 환경에서 강화학습의 두 가지 알고리즘인 **Monte Carlo control**과 **Q-learning**을 구현하고 학습 과정을 비교합니다.

## 1. 분석 전략

### 1.1 교재 코드 기반 구현

- **2장 코드 기반**: GridWorld 환경, Agent 위치/행동 구조, Monte Carlo return 계산, Q-learning TD 갱신식을 사용했습니다.
- **4장 코드 기반 비교 요소**: player/agent 클래스 구조, `epsilon-greedy` 정책, 학습 루프에서 reward/step/fall/success를 기록하고 두 알고리즘을 비교하는 방식을 반영했습니다.
- 신경망/Keras는 사용하지 않고, 상태 수가 작은 Cliff Walking 문제에 적합한 **Q-table 기반 구현**을 사용합니다.

### 1.2 알고리즘

- **Monte Carlo control**: 에피소드가 끝난 뒤 저장된 `(state, action, reward)` 기록을 역순으로 보며 return `G`를 계산하고, 방문 횟수 기반 평균으로 `Q(s,a)`를 갱신합니다. 2장 Monte Carlo 방식처럼 exploring starts를 사용해 다양한 상태-행동 쌍을 방문합니다.
- **Q-learning**: 매 step마다 `reward + gamma * max Q(next_state, action)` TD target을 사용해 `Q(s,a)`를 즉시 갱신합니다.

Cliff Walking 환경은 절벽 보상 `-100`, 도착 보상 `+100`, 일반 이동 보상 `-1`로 설정했습니다. 절벽에 빠지면 시작점으로 돌아가고, 목표에 도달했을 때만 에피소드가 종료됩니다.

### 1.3 평가 지표

- **Train Reward / Train Steps**: 마지막 50개 학습 에피소드의 평균 보상과 평균 step 수.
- **Train Falls**: 학습 중 절벽에 빠진 총 횟수.
- **Train Success**: 학습 중 목표 지점에 도달한 비율.
- **Eval Reward / Eval Steps / Eval Success**: 학습 후 `epsilon=0` greedy 정책으로 별도 평가한 결과.

## 2. 설치 및 실행 방법

### 2.1 환경 구축

```bash
pip install -r environment.txt
```

### 2.2 파이썬 스크립트 실행

```bash
python compare_training.py
```

실행 후 콘솔에 성능 요약 표가 출력되고, 그래프는 아래 경로에 저장됩니다.

```text
팀플/learning_comparison_mc_vs_ql.png
팀플/monte_carlo_greedy_path.png
팀플/q_learning_greedy_path.png
```

### 2.3 주피터 노트북 실행

- 파일: `RL_Comparison.ipynb`
- 노트북은 `environment.py`, `MonteCarlo/mc_agent.py`, `QLearning/q_agent.py`, `compare_training.py`를 import해서 동일한 실험을 실행합니다.

## 3. 결과물

- **`팀플/learning_comparison_mc_vs_ql.png`**: Monte Carlo와 Q-learning의 학습 보상, step 수, 누적 성공률, 누적 절벽 추락 횟수 비교 그래프.
- **`팀플/monte_carlo_greedy_path.png`**: Monte Carlo 학습 후 greedy 정책의 실제 이동 경로.
- **`팀플/q_learning_greedy_path.png`**: Q-learning 학습 후 greedy 정책의 실제 이동 경로.
- **콘솔/노트북 출력 표**: 두 알고리즘의 학습 후반부 성능과 greedy 평가 성능 비교 표.
