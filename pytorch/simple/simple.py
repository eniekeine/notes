import torch
import torch.nn as nn
import torch.optim as optim

class Single(nn.Module): # 나만의 모델을 정의하겠습니다
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.])) # w를 일단 1로 정하겠습니다. 나중에 학습하면서 서서히 고치겠습니다.
        self.bias = nn.Parameter(torch.tensor([1.])) # b를 일단 1로 정합니다. 나중에 학습하면서 서서히 고치겠습니다.
    def forward(self, input):
        return self.weight * input + self.bias # 내 모델은 문제의 해답이 일차 함수라는 가설입니다. 기울기는 weight, y절편은 bias 입니다.

model = Single()
input = torch.tensor([10.0]) # 알려진 입력입니다.
target = torch.tensor([12.0]) # 알려진 출력입니다.
criterion = nn.MSELoss() # 오류는 평균 제곱으로 측정하겠습니다.
learning_rate = 0.001 # 학습률입니다.
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 스토캐스틱 그래디언트 디센트 전략을 사용해 학습하겠습니다.
max_epoch = 1000 # 10000번 학습해도 안 되면 포기하겠습니다.
losses = [] # 시기 별 로스를 기록하겠습니다.
for i in range(max_epoch):
    predict = model.forward(10) # 현재 모델이 예측을 합니다.
    optimizer.zero_grad() # 이전 루프에서 계산했던 편미분을 모두 초기화 하겠습니다.
    loss = criterion(predict, target) # 예측과 타겟 사이의 오류를 검사합니다
    print(f"epoch {i+1} - output = {predict.item()}, loss = {loss.item()}") # 몇 번째 학습인지 표시합니다.
    losses.append(loss.item()) # 시기 별 로스를 기록하겠습니다.
    if loss.item() < 1.e-10: # 오차가 엄청 작으면 그냥 학습을 그만두겠습니다
        break
    else: # 예측과 타겟에 오류가 있었다면
        loss.backward() # 파라미터들에게 오류를 전파하고, 파라미터에 대한 오류의 편미분을 계산합니다.
        optimizer.step() # 스토캐스틱 그래디언트 디센트 전략을 통해 계산한 편미분을 참고하여 파라미터를 조절하겠습니다.
print(model)

import matplotlib.pyplot as plt
import math
x = range(1, i+2)
y = [math.log(loss) for loss in losses]
plt.gca().set_ylim(-30, 100)
plt.title('epoch to loss graph', fontsize=10)
plt.xlabel('epoch', fontsize=8)
plt.ylabel('log(loss)', fontsize=8)
plt.plot(x, y)
plt.show()