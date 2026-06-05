import numpy as np
import copy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class MCAgent:
    """
    몬테카를로(Monte Carlo) 에이전트
    에피소드가 종료된 후 수집된 경험(G_t)을 바탕으로 신경망을 학습합니다.
    """
    def __init__(self, state_size=48, action_size=4, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size # 4x12 = 48 states
        self.action_size = action_size # 상, 하, 좌, 우
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # environment.py의 move() 메서드와 호환을 위한 속성
        self.pos = [3, 0]
        self.action = np.array([[-1,0],[0,1],[1,0],[0,-1]]) # 위, 오른쪽, 아래, 왼쪽 (2장코드 기준)
        
        # 신경망 모델 생성 (RL_Comparison_Strategy.md 권장 구조)
        self.model = self._build_model()
        
        # 에피소드 저장을 위한 메모리
        self.memory = []

    def _build_model(self):
        """동일한 구조의 DNN 설계 (Dense 레이어 2~3개)"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
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
        
        q_values = self.model.predict(state_onehot)
        return np.argmax(q_values[0])

    def append_sample(self, state, action, reward):
        """에피소드 중 발생하는 샘플 저장"""
        self.memory.append((state, action, reward))

    def train_model(self):
        """
        에피소드 종료 후 학습 (Monte Carlo Update)
        G_t = r_t + gamma * r_{t+1} + ...
        """
        G = 0
        states = []
        targets = []
        
        # 에피소드의 끝에서부터 역순으로 Return 계산
        for i in reversed(range(len(self.memory))):
            state, action, reward = self.memory[i]
            G = reward + self.gamma * G
            
            state_onehot = self.state_to_onehot(state)
            target = self.model.predict(state_onehot)
            target[0][action] = G # 해당 행동의 타겟을 G_t로 설정
            
            states.append(state_onehot[0])
            targets.append(target[0])
            
        # 모델 업데이트
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        # 메모리 초기화
        self.memory = []
        
        # 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
