import numpy as np


class MCAgent:
    """
    몬테카를로 에이전트.
    에피소드를 끝까지 가보고 각 (state, action)에 대해 return G를 incremental average로 계산한다. 
    이후 평균낸 G를 바탕으로 Q테이블을 업데이트한다. 
    """
    def __init__(
        self,
        state_shape=(4, 12),
        action_size=4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        first_visit=True,
        **_,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.first_visit = first_visit

        # environment.py의 move() 메서드와 호환되는 2장 Agent 구조
        self.pos = np.array([3, 0])
        self.action = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.q_table = np.zeros((*self.state_shape, self.action_size))
        self.q_visit = np.zeros((*self.state_shape, self.action_size))
        self.memory = []

    def set_pos(self, position):
        self.pos = np.array(position)
        return self.pos

    def get_pos(self):
        return self.pos

    def select_action(self, state=None):
        """2장/4장 코드의 epsilon-greedy 방식으로 행동 선택"""
        pos = self.pos if state is None else np.array(state)

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.q_table[pos[0], pos[1], :]
        max_actions = np.flatnonzero(q_values == np.max(q_values))
        return np.random.choice(max_actions)

    def append_sample(self, state, action, reward):
        self.memory.append((np.array(state), action, reward))

    def train_model(self):
        """
        2장 Monte Carlo control:
        Q(s,a) <- average(Return(s,a))
        """
        G = 0
        visited = set()

        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            key = (int(state[0]), int(state[1]), int(action))

            if self.first_visit and key in visited:
                continue

            visited.add(key)
            row, col, act = key
            self.q_visit[row, col, act] += 1
            self.q_table[row, col, act] += (
                (G - self.q_table[row, col, act]) / self.q_visit[row, col, act]
            )

        self.memory = []
        self.decay_epsilon()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
