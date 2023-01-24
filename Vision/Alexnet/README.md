# ImageNet Classification with Deep Convolutional Neural Networks

## Abstract

본 논문에서 제안한 모델인 Alexnet은 12만 개의 고해상도 이미지 1,000개의 클래스로 분류하는 ImageNet LSVRC-2010 대회에서 기존의 최첨단 모델보다 좋은 성능인 top-1, top-5에서 각각 오차율을 37.5%, 17%를 달성하였다.  Alexnet은 6천만 개의 파라미터와 65만 개의 뉴런, max-pooling을 동반한 5개의 CNN과 3개의 Fully-Connected Layers로 이루어져 있다. 또한 마지막 Fully-Connected Layer에는 1000-way softmax를 적용한다.  본 연구에서 학습 속도 향상을 위해 제안한 방법으로는 GPU 병렬 처리 기법과 non-saturating neurons(본 논문에서는 ReLU 사용)을 사용했다. 또한 과적합 문제를 막기 위해 dropout 기법을 사용하였다.

<br>

## Introduction

시간이 지날수록 분류해야 할 이미지 데이터 셋의 크기, 카테고리 수는 커지고 해상도 또한 높아지고 있는 추세에서 기존의 순방향 신경망보다 CNN은 비교적 더 적은 연결과 파라미터 수로 쉽게 학습이 가능하며, 성능 또한 우수하다. 하지만 CNN에 큰 규모의 고해상도 이미지를 적용하기에는 어려움이 존재하며 이를 GPU 병렬 처리로 해결하였다.

<br>

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

<br>

## The Architecture

본 논문에서 모델 구조를 설명하기 전 제안한 신경망의 몇 가지 특징들을 설명하며, 그 특징들은 아래와 같다.

```
1. ReLu Nonlinearity

2. Training on Multiple GPUs

3. Local Response Normalization

4. Overlapping Pooling
```

<br>

### 1. ReLu Nonlinearity

일반적으로 자주 쓰이며 saturating nonlinearity인 Hyperbolic Function이나 Sigmoid Function 대신 non-saturating nonlinearity인 ReLU Function을 채택하였다. 그 이유는 규모가 큰 데이터 셋을 학습 시킬시 동일 오차에 다다르는 속도가 우수하기 때문이다. 

<br>

### 2. Training on Multiple GPUs

본 연구에 사용된 GTX 580 GPU는 3GB 밖에 메모리가 없기 때문 하나의 network를 두개의 GPU에 나눠서 학습을 진행하였다. GPU parallelization은 커널을 반으로 나누어 각각 하나의 GPU에 할당하는 것이며 추가로 두 GPU 간의 communication은 특정 layer에서만 발생하도록 하였다. 결과적으로 layer 3에서는 layer 2의 모든 커널 map을 받아 올 수 있지만 layer 4는 같은 GPU의 입력만 받아온다.
이러한 병렬 처리는 하나의 GPU에서 훈련 된 것과 비교하면 top-1과 top-5에서 각각 오차율을 1.7% 및 1.2% 줄였으며, 2-GPU net은 1-GPU net2보다 학습 시간을 줄인다.

<br>

### 3. Local Response Noramlization

ReLU Function은 saturation를 방지하기 위해 입력 정규화를 진행할 필요가 없으나, Local Response Normalization는 일반화에 도움이 된다. Local Response Normalization은 측면 억제(lateral inhibition)을 구현한 형태로 타 커널에서 계산된 출력과 경쟁을 일으키는 것이다.

본 내용은 측면 억제의 대표적 예시인 헤르만 격자를 통해 이해해보도록 하겠다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/97859215/214345170-290b9d03-4742-433c-86c8-ec59e34332f0.png">
</p> 

위 사진은 헤르만 격자로 검은 사각형 안에 흰색의 선들이 교차되고 있다. 흰색의 선에 집중하지 않을 때, 회색의 점이 보이고 이러한 현상을 측면 억제라 한다. 이러한 현상은 흰색으로 둘러 쌓인 측면에서 억제를 발생시키기에 흰색이 더욱 반감되기 때문이다.

<br>

이렇게 알아본 측면 억제를 본 연구에서 사용한 이유는 ReLU를 채택했기 때문이다.
ReLU는 양수 방향으로는 입력 값을 그대로 출력하기 때문에, 합성곱이나 풀링 시 매우 높은 하나의 픽셀값이 주변 픽셀에 영향을 미치는 문제는 방지하고자 Activation Map의 같은 위치에 존재하는 픽셀끼리 정규화하는 것이다.

<br>

### 4. Overlapping Pooling

일반적으로 CNN을 풀링할 때 풀링되는 뉴런들이 중복되지 않게 풀링을 진행하는데, 본 연구에서는 풀링되는 뉴런들을 중복되도록 진행하여 오차율 감소와 과적합 방지라는 효과를 얻을 수 있었다.

### Overall Architecture

![overall_arch](https://user-images.githubusercontent.com/97859215/214348948-f498eaef-7431-4ea8-b2a4-11a467ba2748.jpg)

Code
```
TBA
```

<br>

## Reducing Overfitting

본 논문에서 제안한 모델인 Alexnet은 파라미터 개수가 6천만 개에 달한다. 저자들이 제한한 과적합 문제를 해결하고자 제한한 두 가지 방법은 아래와 같다.

```
1. Data Argument

2. Dropout
```

