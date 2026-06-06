import numpy as np

# 4 x 12 미로 환경
class Environment():
    
    # 1. 미로밖(절벽), 길, 목적지와 보상 설정
    cliff = -100
    road = -1
    goal = 100
    
    # 2. 목적지 좌표 설정 (우측 하단)
    goal_position = [3,11]
    
    # 2.1 출발 지점 좌표 설정 (좌측 하단)
    start_position = [3,0]
    
    # 3. 보상 리스트 숫자
    reward_list = [[road,road,road,road,road,road,road,road,road,road,road,road],
                   [road,road,road,road,road,road,road,road,road,road,road,road],
                   [road,road,road,road,road,road,road,road,road,road,road,road],
                   [road,cliff,cliff,cliff,cliff,cliff,cliff,cliff,cliff,cliff,cliff,goal]]
    
    # 4. 보상 리스트 문자
    reward_list1 = [["road","road","road","road","road","road","road","road","road","road","road","road"],
                    ["road","road","road","road","road","road","road","road","road","road","road","road"],
                    ["road","road","road","road","road","road","road","road","road","road","road","road"],
                    ["road","cliff","cliff","cliff","cliff","cliff","cliff","cliff","cliff","cliff","cliff","goal"]]
    
    # 5. 보상 리스트를 array로 설정
    def __init__(self):
        self.reward = np.asarray(self.reward_list)    

    # 6. 선택된 에이전트의 행동 결과 반환
    def move(self, agent, action):
        
        done = False
        
        # 6.1 행동에 따른 좌표 구하기
        new_pos = agent.pos + agent.action[action]

        # 6.2 이동 후 좌표가 미로 밖이면 제자리에 머무름
        if (new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or 
            new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]):
            observation = agent.set_pos(agent.pos)
            reward = self.road

        # 6.3 절벽이면 큰 패널티를 받고 시작점으로 돌아가되 에피소드는 계속 진행
        elif self.reward_list1[new_pos[0]][new_pos[1]] == "cliff":
            reward = self.cliff
            observation = agent.set_pos(self.start_position)
            done = False

        # 6.4 목적지에 도착하면 에피소드 종료
        elif self.reward_list1[new_pos[0]][new_pos[1]] == "goal":
            reward = self.goal
            observation = agent.set_pos(new_pos)
            done = True
            
        # 6.5 이동 후 좌표가 길이라면
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0],observation[1]]
            
        return observation, reward, done
