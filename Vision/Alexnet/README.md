# ImageNet Classification with Deep Convolutional Neural Networks

[Research Paper link](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## Abstract

본 논문에서 제안한 모델인 Alexnet은 12만 개의 고해상도 이미지 1,000개의 클래스로 분류하는 ImageNet LSVRC-2010 대회에서 기존의 최첨단 모델보다 좋은 성능인 top-1, top-5에서 각각 오차율을 37.5%, 17%를 달성하였다.  Alexnet은 6천만 개의 파라미터와 65만 개의 뉴런, max-pooling을 동반한 5개의 CNN과 3개의 Fully-Connected Layers로 이루어져 있다. 또한 마지막 Fully-Connected Layer에는 1000-way softmax를 적용한다.  본 연구에서 학습 속도 향상을 위해 제안한 방법으로는 GPU 병렬 처리 기법과 non-saturating neurons(본 논문에서는 ReLU 사용)을 사용했다. 또한 과적합 문제를 막기 위해 Dropout 기법을 사용하였다.

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

ReLU Function은 saturation를 방지하기 위해 입력 정규화를 진행할 필요가 없으나, Local Response Normalization는 일반화에 도움이 된다. 

<br>

-------------------------
<p align="center">Local Response Noramlization 이해를 돕기 위한 추가적인 내용</p>

<br>

Local Response Normalization은 측면 억제(lateral inhibition)을 구현한 형태로 타 커널에서 계산된 출력과 경쟁을 일으키는 것이다.

본 내용은 측면 억제의 대표적 예시인 헤르만 격자를 통해 이해해보도록 하겠다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/97859215/214345170-290b9d03-4742-433c-86c8-ec59e34332f0.png">
</p> 

위 사진은 헤르만 격자로 검은 사각형 안에 흰색의 선들이 교차되고 있다. 흰색의 선에 집중하지 않을 때, 회색의 점이 보이고 이러한 현상을 측면 억제라 한다. 이러한 현상은 흰색으로 둘러 쌓인 측면에서 억제를 발생시키기에 흰색이 더욱 반감되기 때문이다.

---------------------------

<br>

이렇게 알아본 측면 억제를 본 연구에서 사용한 이유는 ReLU를 채택했기 때문이다.
ReLU는 양수 방향으로는 입력 값을 그대로 출력하기 때문에, 합성곱이나 풀링 시 매우 높은 하나의 픽셀값이 주변 픽셀에 영향을 미치는 문제는 방지하고자 Activation Map의 같은 위치에 존재하는 픽셀끼리 정규화하는 것이다.

추가적으로 본 연구 당시에는 Batch Normalization이 없었다고 한다.

<br>

### 4. Overlapping Pooling

일반적으로 CNN을 풀링할 때 풀링되는 뉴런들이 중복되지 않게 풀링을 진행하는데, 본 연구에서는 풀링되는 뉴런들을 중복되도록 진행하여 오차율 감소와 과적합 방지라는 효과를 얻을 수 있었다.

<br>

### Overall Architecture

![overall_arch](https://user-images.githubusercontent.com/97859215/214348948-f498eaef-7431-4ea8-b2a4-11a467ba2748.jpg)

<p align="center">
<img src = "https://user-images.githubusercontent.com/97859215/214492922-5afaac35-55cf-4644-8f1b-1a7b2806ec27.png" width="600" height = "700">
</p>



<br>

__Alexnet__
- Input : 224 x 224 x 3 = 150,528
- Convolution 1 : 11x11 kernel, 4 stride : 54x54x96
- Max pooling 1 : 3x3 kernel, 2 stride : 26x26x96
- Convolution 2 : 5x5 kernel, 2 padding : 26x26x256
- Max pooling 2 : 3x3 kernel, 2 stride : 12x12x256
- Convolution 3 : 3x3 kernel, 1 padding : 12x12x384
- Convolution 4 : 3x3 kernel, 1 padding : 12x12x384
- Convolution 5 : 3x3 kernel, 1 padding : 12x12x384
- Max pooling 3 : 3x3 kernel, 2 stride : 5x5x256
- Dense 1 : 4096
- Dense 2 : 4096
- Dense 3 : 1000

<br>

## Reducing Overfitting

본 논문에서 제안한 모델인 Alexnet은 파라미터 개수가 6천만 개에 달한다. 저자들이 제한한 과적합 문제를 해결하고자 제한한 두 가지 방법은 아래와 같다.

```
1. Data Argument

2. Dropout
```

<br>

### 1. Data Argument

학습 데이터를 인위적으로 변환하여 훈련 데이터를 증가시키는 방법이다. 

<br>

변환된 이미지를 저장하지 않고 GPU 학습시에 CPU에서 계산하도록 하여, 추가적인 계산 비용을 소모하지 않았다. 

<br>

주요 방법은 아래 두 가지로 요약된다.

<br>

1. 256 × 256 이미지에서 224 × 224 패치를 추출 후 수평 방향으로 뒤집기
    * 기존 데이터 셋의 2048 배 확장 가능
    *  5개의 224 × 224 패치 (코너 패치 4개 및 중앙 패치 1개)와 수평 반사를 수행한 10개의 패치 사용

<br>

2. RGB 채널 강도 조정
    - 학습 데이터셋의 픽셀값으로 PCA 수행
    - PCA eigenvector에 N(0,0.1)인 정규분포에 추출한 랜덤값을 곱해 색상을 조정
    - Top-1 오차율을 1% 감소

<br>

### 2. Dropout

또 다른 과적합 방지 방법으로 Dense Layer의 Output에 Dropout rate = 0.5를 사용한 Dropout layer를 추가한다. 학습 시 Epoch을 2배 이상 늘렸음에도, 과적합을 성공적으로 방지했음을 알 수 있었다.

<br>

## Details of learning

- Batch Size : 128
- SGD (momentum : 0.9, weight decay : 0.0005)
    - weight decay가 모델 정규화 뿐만 아니라 모델의 학습 오차 또한 감소시켰다.
- Weight Initialize
    - 평균 : 0, 표즌 편차 : 0.01인 정규 분포 따르도록 초기화
    - 두 번째, 네 번째, 다섯 번째 convolution과 dense layer의 편향은 1로 초기화하여 학습 가속화 효과
- Learning Rate
    - 모든 layer에 대해 동일, but 훈련 수행 중 매뉴얼하게 조정
    - LR : 0.01 시작 -> 학습 개선 X -> LR = LR / 10

<br>

## Result

<p align="center">
<img src = "https://user-images.githubusercontent.com/97859215/214496451-c1a50793-734e-49e5-ab9d-2546ce9efa9b.png" width="500" height = "300">
</p>

ILSVRC-2010 데이터에 대해서 기존 모델이 도출한 결과를 압도하는 결과 제시

<p align="center">
<img src = "https://user-images.githubusercontent.com/97859215/214496712-55c6f102-952e-43a2-adb1-09362837aeb1.png" width="700" height = "300">
</p>

본 결과는 CNN Layer가 많아질수록 오차율이 줄어드는 것을 보여준다.


## Qualitative Evaluations

<p align="center">
<img src = "https://user-images.githubusercontent.com/97859215/214498244-b4b1c777-1c6b-4537-a8ec-4f5130d3e159.png" width="300" height = "300">
</p>

<p align="center">
<img src = "https://user-images.githubusercontent.com/97859215/214497908-1ad7a460-993f-4ea4-a849-398e7fcda10d.png" width="700" height = "500">
</p>



CNN Kernel을 시각화한 Figure 3을 통해, 각 Kernel이 이미지의 다양한 Feature를 효과적으로 추출했음을 보여준다.

<br>

Figure 4를 통해 본 논문에서 제안한 Alexnet은 중앙에서 벗어나는 이미지 데이터도 효과적으로 분류해냈음과 Top-5 예측이 대부분 유사한 범주임을 보여주어 합리적인 예측을 수행하고 있음 또한 보여준다.

<br>

추가적으로 자세가 서로 다른 코끼리의 사례처럼 Pixel 차원에서 완전히 다른 데이터임에도 유사한 범주로 분류할 수 있는 결과를 보여준다.


## Discussion

CNN Layer를 쌓을수록 효과적으로 작동한 결과를 보여주었다. 즉, "Deep"한 CNN이 오차율 감소에 중요하다는 것을 강조하며 역으로 CNN Layer를 제거할 때마다 Top-1 Accuracy 가 2%씩 감소한다.