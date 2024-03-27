# Rethinking the Inception Architecture for Computer Vision

## Abstract

<br>

본 연구에서는 적절한 Convolution 분해와 과감한 Regualarization을 통해 추가된 연산량을 효율적으로 사용하는 것에 초점을 둔 Network 크기 증가 방법에 대해 탐구한다.

<br>
<br>

## Introduction

<br>

분류 부문에서 성능이 높아질수록 다른 분야에 응용했을 때도 성능이 높아진다는 사실을 발견했다.

이는 개선된 Convolution Architecture는 Architecture 자체의 성능이 발전하면 이걸 컴퓨터 비전의 대부분의 분야, 즉 고품질의 학습된 시각적 특징에 의존하는 분야들에서도 성능이 발전한다는 것이다.

Inception의 연산량은 VGGNet 또는 그 보다 더 높은 성능을 갖는 모델들보다 훨씬 적다.

이러한 점은 Inception을 한정된 메모리 혹은 계산량이 내포된 상황에서 상당한 양의 데이터를 처리해야 하는 빅데이터 상황에서의 사용을 쉽게 할 수 있게 해준다.

허나 Inception Architecture의 복잡한 구조는 Network를 변경시키는데 어려움을 겪게 한다.

만약 Architecture가 단순하게 커진다면, 계산에서 얻은 이점들을 사라지게 할 것이다.

<br>
<br>

## General Design Principles

<br>

본 파트에서 설명하는 원칙들을 벗어나면 Network 성능은 떨어지고, 원칙들을 충족하면 Architecture가 일반적으로 개선되는 경향을 보인다고 한다.

```
1. Representational Bottleneck 피하기, 특히 네트워크 초기 부분에서

(-> Representational Bottleneck이란, pooling으로 인해 feature map의 size가 감소하면서 정보량이 감소하는 것)

2. 고차원 Representation은 Network 내에서 지역적으로 쉽게 처리 가능

(-> 즉 Convolution을 잘게 쪼개서 activation을 많이 할수록 feature를 더 잘 학습한다는 뜻)

3. 공간 집합은 저수준 차원에서의 임베딩은 Representational Porwer를 많이 잃지 않으면서 적용 가능

(-> 정리하자면, Convolution 전 차원 축소를 해도 인접한 Unit의 상관관계로 인해 정보 손실이 적어 큰 문제가 없고, 학습도 빨라진다는 뜻)

4. Network의 깊이와 너비의 밸런스 맞추기
```

<br>
<br>

## Factorizing Convolutions with Large Size

<br>

본 논문에서는 여러 세팅에서의 Convolutions 분해 방법에 대해 탐구하며, 특히 Computational 효율성을 높이기 위함에 초점을 둔다.

적절한 분해는 파라미터에 구애를 덜 받으며 빠른 속도의 학습을 가능케하며 절감한 계산 비용과 메모리로 각 모델의 복제품을 단일 컴퓨터에서 훈련하는 능력을 유지하고 Network의 필터 뱅크 크기를 키우는데 사용 가능하다.

<br>
<br>

## Factorization into smaller convolutions

<br>

해당 파트에서는 5x5 Convolution을 3x3 Convolution 두 개로 대체하는 방법에 대해 서술한다.

먼저 5x5 Convolution을 3x3 Convolution 두 개를 사용했을 때의 출력값은 같고, 파라미터 수는 5x5 = 25, 2x(3x3) = 18로 28%의 연산량 이득을 취할 수 있다고 한다.

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/068c9c3d-dc21-4bda-a1e9-92b96c0ab08e">
</p> 

<br>

```
(1) 3x3 -> Linear -> 3x3 -> ReLU

(2) 3x3 -> ReLU -> 3X3 -> ReLU
```

다음으로는 위 두 개의 구조를 비교한다.

실험을 통해 Linear Activation이 항상 ReLU보다 낮은 성능을 보여줌을 알 수 있었다고 한다. 

이러한 이득은 네트워크가 학습할 수 있는 다양하게 늘어난 공간, 특히 activation의 출력에 batch normalize를 하는 곳에서 더욱 얻게 한다.

이는 차원 축소를 위한 Linear Activation을 사용했을 때도 유사한 효과를 얻을 수 있다고 한다.


<br>
<br>

## Spatial Factorization into Asymetric Convolutions

해당 파트에서는 3x3 Convolution 하나와 3x1 Convolution과 1x3 Convolution, 두 레이어로 대체하는 방법에 대해 서술한다.

3x3 Convolution 하나를 썼을 때와 3x1, 1x3 Convolution을 사용했을 때, 33% 의 계산 비용을 절약할 수 있는 반면 3x3 Convolution 하나와 2x2 Convolution 두 개를 썼을 때는 11% 밖에 절약시키지 못한다.

해당 개념을 확장하며 nxn Convolution을 nx1, 1xn 두 개의 레이어로 대체하며 n의 값이 커질수록 Computational Cost 절약은 급격하게 상승한다.

실험에 의하면 mxm Feature Map에서 12<m<20 일 때, 좋은 성능을 보였으며, 초기 layer에서는 그다지 좋은 성능을 내지 못하는 것을 확인할 수 있었다고 한다.

또한 그 레벨에서는 1x7 + 7x1이 매우 좋은 성능을 보인다고 한다.


<br>
<br>

## Utility of Auxiliary Classifiers

<br>

연구진들은 Auxiliary Classifiers가 학습 초기에는 수렴이 개선되지 않는 결과를 확인하였고, 학습 후반에는 정확도 개선에 좋은 영향을 주는 것을 확인하였다고 한다.

낮은 층의 Auxiliary Classifier는 제거해도 Network의 최종 결과에 영향이 없다고 한다.

그들은 Auxiliary Classifires를 Regularizer로 사용하기로 한다.

이는 Side Branch에 Batch-Normalized 또는 Dropout Layer를 두었을 때, 네트워크의 Main Classifier의 성능이 더 좋아진다는 사실에 기반을 두었다.

또한 이는 Batch Normalization이 Regularizer와 같은 역할을 한다는 추측에 약간의 힘을 실어준다.

<br>
<br>

## Efficient Grid Size Reduction

<br>

```
dxd grid & k filter => 2 x (d^2) x (k^2)

(d/2)x(d/2) grid & 2k filter => 2 x (d/2)^2 x k^2


=> 연산량은 줄지만 표현력 또한 감소된다. 따라서 연산량은 줄고 표현력은 챙기는 Figure 10. 구조를 제안한다.

```

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/3314f686-cf6f-4297-ad18-5b2550f20fe3">
</p> 

<br>
<br>


## Inception-v2

<br>

To be Continue....
