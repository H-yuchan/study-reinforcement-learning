import numpy as np
from environment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0,0] * env.width for _ in range(env.height)]
        # 상하좌우 동일한 확률로 정책 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]

        #마침 상태의 설정
        self.policy_table[2][2] = []
        # 할인율
        self.discount_factor = 0.9

    # 벨만 기대 방정식을 통해 다음 가치함수를 계산하는 정책 평가
    def policy_evaluation(self):
        # 다음 가치함수 초기화
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        # 모든 상태의 대해서 벨만 기대 방정식 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태의 가치함수 = 0
            if state == [2,2]:
                next_value_table[state[0]][state[1]] = value
                continue
            #벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value+= (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))

