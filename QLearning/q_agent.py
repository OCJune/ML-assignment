import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class QLearningAgent:
    """
    Q-러닝(Q-Learning) 에이전트
    매 스텝마다 TD 타겟(r + gamma * max Q')을 이용해 신경망을 즉시 학습합니다.
    """
    def __init__(self, state_size=48, action_size=4, learning_rate=0.001, gamma=0.99, hidden_layers=[64, 64]):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.hidden_layers = hidden_layers
        
        # environment.py의 move() 메서드와 호환을 위한 속성
        self.pos = [3, 0]
        self.action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
        
        # 신경망 모델 생성
        self.model = self._build_model()

    def _build_model(self):
        """
        1. 알고리즘: 예제코드(2장)의 Q-러닝 방식을 확장하여 DNN을 함수 근사기로 사용했습니다.
        2. 기법 간소화: 예제코드(4장)의 DQN과 달리, MC와의 순수한 비교를 위해 리플레이 버퍼와 타겟 네트워크를 생략하고 매 스텝 즉시 학습합니다.
        3. 입력 데이터: Tic Tac Toe(Conv2D)와 달리, 그리드 상태를 원-핫 인코딩(1x48)하여 Dense 레이어로 처리합니다.
        4. 유연한 아키텍처: hidden_layers 파라미터로 모델 구조를 동적으로 변경하여 성능 분석이 가능하게 설계했습니다.
        """
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        for units in self.hidden_layers:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def set_pos(self, position):
        """에이전트의 위치를 설정 (environment.py 호환)"""
        self.pos = np.array(position)
        return self.pos

    def get_pos(self):
        """에이전트의 현재 위치를 반환"""
        return self.pos

    def state_to_onehot(self, pos):
        """좌표 [r, c]를 48차원 원-핫 벡터로 변환"""
        onehot = np.zeros(self.state_size)
        index = pos[0] * 12 + pos[1]
        onehot[index] = 1.0
        return np.reshape(onehot, [1, self.state_size])

    def select_action(self, state_onehot):
        """엡실론-그리디 전략에 따른 행동 선택"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self.model.predict(state_onehot, verbose=0)
        return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, done):
        """
        매 스텝마다 학습 (Q-Learning Update)
        Target = r + gamma * max Q(s', a')
        """
        state_onehot = self.state_to_onehot(state)
        next_state_onehot = self.state_to_onehot(next_state)
        
        target = self.model.predict(state_onehot, verbose=0)
        
        if done:
            target[0][action] = reward
        else:
            # 다음 상태에서의 최대 Q값 탐색 (Off-policy)
            next_q_values = self.model.predict(next_state_onehot, verbose=0)
            target[0][action] = reward + self.gamma * np.max(next_q_values[0])
            
        # 모델 업데이트 (즉시 학습)
        self.model.fit(state_onehot, target, epochs=1, verbose=0)
        
        # 에피소드 종료 시 엡실론 감소
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
