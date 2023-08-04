# 참고 : https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975
import numpy as np
import random

# 활성화 함수
def activation(weighted_sum):
    if weighted_sum >= 0 : return 1
    else : return 0
# 가능한 모든 입력. 모든 입력의 첫번 째 값이 1인 이유는 바이어스(ws[0])에 곱해지는 부분이기 때문이다.
inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# OR, AND, XOR 연산에 대한 정답
or_answers = np.array([0, 1, 1, 1])
and_answers = np.array([0, 0, 0, 1])
xor_answers = np.array([0, 1, 1, 0])
# 퍼셉트론이 모든 가능한 입력에 대해 정답을 낼 경우 훈련이 완료됨.(convergence)
def check_convergence(ws, answers):
    # 모든 입력에 대한 각각의 가중합을 구한다
    weighted_sums = [np.dot(input, ws) for input in inputs]
    # 활성화 함수에 각 가중합을 입력한 결과를 저장한다
    expectations =  np.vectorize(activation)(weighted_sums)
    # 정답과 예측이 서로 일치하면 True를 반환한다.
    return np.array_equal(expectations, answers)
# 훈련시키기
def train(answers, loop_limit = 1024):
    # 가중치. 최초에는 랜덤 값을 넣어둔다.
    ws = np.random.uniform(0, 1, 3)
    # 루프 횟수를 헤아리기 위한 변수.
    loop_count = 0
    # 훈련이 완료될 떄 까지
    while check_convergence(ws, answers) == False and loop_count < loop_limit:
        # 훈련 루프 횟수를 헤아리고
        loop_count = loop_count + 1
        # 랜덤하게 네 가지 데이터중 i 번째 데이터를 고른다.
        i = random.randint(0, 3)
        # i 번째 입력을 가져온다.
        xs = inputs[i]
        # i 번째 정답을 가져온다.
        answer = answers[i]
        # 가중합을 계산한다.
        weighted_sum = np.dot(xs, ws)
        # 정답이 1인데 0을 예측한 경우
        if answer == 1 and activation(weighted_sum) == 0:
            ws = ws + xs # 가중치를 조정한다
        # 정답이 0인데 1을 예측한 경우
        elif answer == 0 and activation(weighted_sum) == 1:
            ws = ws - xs # 가중치를 조정한다
    return ws, loop_count, check_convergence(ws, answers)

ws, loop_count, success = train(or_answers)
print('ws : ', ws, 'loop : ', loop_count, 'Success' if success else 'Failed')
ws, loop_count, success = train(and_answers)
print('ws : ', ws, 'loop : ', loop_count, 'Success' if success else 'Failed')
# Linearly Separable하지 않기 때문에 이 훈련은 실패해야 정상이다.
ws, loop_count, success = train(xor_answers)
print('ws : ', ws, 'loop : ', loop_count, 'Success' if success else 'Failed')