# ImageNet Classification with Deep Convolutional Neural Networks

## Abstract

본 논문에서 제안한 모델인 Alexnet은 12만 개의 고해상도 이미지 1,000개의 클래스로 분류하는 ImageNet LSVRC-2010 대회에서 기존의 최첨단 모델보다 좋은 성능인 top-1, top-5에서 각각 37.5%, 17%를 달성하였다.  Alexnet은 6천만 개의 파라미터와 65만 개의 뉴런, max-pooling을 동반한 5개의 CNN과 3개의 Fully-Connected Layers로 이루어져 있다. 또한 마지막 Fully-Connected Layer에는 1000-way softmax를 적용한다.  본 연구에서 학습 속도 향상을 위해 제안한 방법으로는 GPU 병렬 처리 기법과 non-saturating neurons(본 논문에서는 ReLU 사용)을 사용했다. 또한 과적합 문제를 막기 위해 dropout 기법을 사용하였다.

## Introduction

시간이 지날수록 분류해야 할 이미지 데이터 셋의 크기, 카테고리 수는 커지고 해상도 또한 높아지고 있는 추세에서 기존의 순방향 신경망보다 CNN은 비교적 더 적은 연결과 파라미터 수로 쉽게 학습이 가능하며, 성능 또한 우수하다. 하지만 CNN에 큰 규모의 고해상도 이미지를 적용하기에는 어려움이 존재하며 이를 GPU 병렬 처리로 해결하였다.

## The Dataset

ImageNet의 이미지 데이터는 다양한 크기를 띄고 있으며, 모델이 요구하는 입력 사이즈에 맞춰주기 위한 작업을 진행하였다. 작업 진행을 아래와 같이 진행되었다.

<br>

![그림1](https://user-images.githubusercontent.com/97859215/214329869-a11f5633-45d7-4e1c-9132-227ef37d51fc.png)

```
1. 직사각형의 Raw Image Data의 짧은 변의 길이를 256 사이즈로 변환

2. 256 * 256 patch를 생성하여 Center Crop 진행

3. 256 * 256 사이즈의 이미지 생성됨
```

또한, 본 연구에서는 각 픽셀에서 훈련 세트의 평균을 뺀 것 이외의 전처리 작업은 진행하지 않았다.

## The Architecture

본 논문에서 모델 구조를 설명하기 전 제안한 신경망의 몇 가지 특징들을 설명하며, 그 특징들은 아래와 같다.

```
1. ReLu Nonlinearity

2. Training on Multiple GPUs

3. Local Response Normalization

4. Overlapping Pooling
```

### 1. ReLu Nonlinearity

일반적으로 자주 쓰이며 saturating nonlinearity인 Hyperbolic Function이나 Sigmoid Function 대신 non-saturating nonlinearity인 ReLU Function을 채택하였다. 그 이유는 규모가 큰 데이터 셋을 학습 시킬시 동일 오차에 다다르는 속도가 우수하기 때문이다. 


### 2. Training on Multiple GPUs

본 연구에 사용된 GTX 580 GPU는 3GB 밖에 메모리가 없기 때문 하나의 network를 두개의 GPU에 나눠서 학습을 진행하였다. GPU parallelization은 커널을 반으로 나누어 각각 하나의 GPU에 할당하는 것이며 추가로 두 GPU 간의 communication은 특정 layer에서만 발생하도록 하였다. 결과적으로 layer 3에서는 layer 2의 모든 커널 map을 받아 올 수 있지만 layer 4는 같은 GPU의 입력만 받아온다.

### 3. Local Response Noramlization



