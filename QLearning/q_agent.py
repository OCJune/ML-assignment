import numpy as np


class QLearningAgent:
    """
    Q-learning 에이전트.

    2장 코드의 TD(0) Q-learning 갱신식을 Cliff Walking 환경에 맞게 옮긴 구현입니다.
    4장의 Q_learning_player처럼 policy와 learn 함수를 분리하되, 상태가 작은
    grid world이므로 신경망 대신 Q-table을 사용합니다.
    """
    def __init__(
        self,
        state_shape=(4, 12),
        action_size=4,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        **_,
    ):
        # 외부에서 하이퍼파라미터를 넘겨받을 수 있도록 매개변수화
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # environment.py의 move() 메서드와 호환되는 2장 Agent 구조
        self.pos = np.array([3, 0])
        self.action = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # Cliff Walking 격자 크기에 맞춰 고정 크기의 3차원 NumPy 배열 정의
        self.q_table = np.zeros((*self.state_shape, self.action_size))

    def set_pos(self, position):
        self.pos = np.array(position)
        return self.pos

    def get_pos(self):
        return self.pos

    def select_action(self, state=None):
        """2장/4장 코드의 epsilon-greedy 방식으로 행동 선택"""
        # 외부에서 state(좌표)를 직접 주입받아 행동 선택 가능
        # np.flatnonzero를 사용하여 최대 Q값을 가진 모든 행동들 중 균등한 확률로 무작위 선택
        pos = self.pos if state is None else np.array(state)

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.q_table[pos[0], pos[1], :]
        max_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(max_actions)

    def train_model(self, state, action, reward, next_state, done):
        """
        2장 Q-learning:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        """
        row, col = int(state[0]), int(state[1])
        next_row, next_col = int(next_state[0]), int(next_state[1])

        now_q = self.q_table[row, col, action]
        next_q = 0 if done else np.max(self.q_table[next_row, next_col, :])
        target = reward + self.gamma * next_q

        self.q_table[row, col, action] += self.learning_rate * (target - now_q)

    def decay_epsilon(self):
        # 매 에피소드 종료 시점마다 지수 형태로 감쇄하여 epsilon_min 하한선까지 decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
