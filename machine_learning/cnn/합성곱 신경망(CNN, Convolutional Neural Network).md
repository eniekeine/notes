# 합성곱 신경망(CNN, Convolutional Neural Network)

- 합성곱 신경망(Convolutional Neural Networks, CNNs)은 인간의 시각 처리 방식을 모방한 신경망으로, 이미지 처리에 특출난 성능을 보인다.
- 합성공 신경망은 Yann LeCun, Pierre Sermanet, Léon Bottou, Yoshua Bengio, Patrick Haffner가 1990년대 에 고안했다.
- 당시 AT&T Labs Bell Labs(추후 AT&T Labs-Research)에서 근무 중이던 Yann LeCun이 개발을 주도했다.
- 이 신경망은 최초로 역전파를 활용하여 손글씨 숫자를 인식하는 인공신경망이다.
- 그 인공신경망은 "LeNet-5"라 이름붙었다.
![LeNet5 모델의 도식](<LeNet5 모델의 도식.png>)
- LeNet-5는 합성곱층(convolution layer)를 사용하여 이차원 행렬화된 손글씨 이미지를 처리한다.
- LeNet-5는 여러 개의 합성곱층, 풀링층(pooling layer)과 마지막에 분류를 위한 전결합층(fully connected layer)으로 구성된다.
- 합성곱 레이어는 피처 추출(feature extraction)을 훈련 가능한 필터(learnable filters) 즉 "커널"(kernel)을 이미지에 적용하여 실시한다.
- 풀링층은 학습된 피처(leanred feature)의 차원수(dimensionality)을 낮추어, 신경망을 안정화하고 필요한 계산량을 줄인다.
- LeNet-5는 현대의 CNN의 시초가 되는 중요한 모델이다.
- CNN은 컴퓨팅 파워의 한계 때문에 한동안 작은 이미지만 분류할 수 있어서 크게 주목받지 못했었다.
- 그런데 2012년 "ImageNet image classification challenge"에서 Krizhevsky의 CNN 모델이 우승하면서 세상에 크게 알려지게 된다.(Krizhevsky et al., 2012)
- CNN은 이미지 분류, 개체 감지, 의미론적 분할 등 다양한 컴퓨터 비전(computer vision) 작업에 활용된다.
- 이후에는 자연어 처리와 음성 인식에도 활용되었다.
- 2012년 근처 부터는 "transposed convolutional layers"(혹은 fractionally strided convolutional layers)를 활용한 CNN 신경망이 유행하기 시작했다. (Zeiler et al., 2011; Zeiler and Fergus, 2014; Long et al., 2015; Radford et al., 2015; Visin et al., 2015; Im et al., 2016)

## 참고

<https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939>
<https://programmathically.com/what-is-a-convolution-introducing-the-convolution-operation-step-by-step/>
<https://arxiv.org/pdf/1603.07285v1.pdf>