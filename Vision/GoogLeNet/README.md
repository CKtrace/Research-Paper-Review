# [Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

## Abstract

본 논문에서는 Deep Convolutional Neural Network Architecture인 Inception을 제안하였다. 본 구조는 네트워크의 계산 자원을 효율적으로 지키며 네트워크의 깊이와 너비를 늘려갈 수 있도록 정교하게 구조화되었다고 한다. 또한 네트워크의 계산 자원의 효율성을 위해 구조 설계는 Hebbian Pricinple과 Multi-Scale Processing를 기반으로 진행되었다. Inception 구조를 이용하여 탄생한 GoogLeNet 모델은 22층을 갖는 신경망이며, ILSVRC14에서 분류와 탐지에 사용하였다고 한다.

<br>

## Introduction

본 논문에서 제안한 모델인 GoogLeNet의 파라미터 수는 2년 전 출시한 Alexnet 모델의 파라미터보다 12배 적으면서 더욱 정확하다고 한다. Object Detection 부분에서는 단순히 깊고 크게 만든 네트워크보다는 R-CNN와 같이 Deep Architecture의 시너지 효과와 Classical Computer Vision 알고리즘으로부터 큰 성능 향상이 이루어졌다.

<br>

또 다른 중요한 요소는 모바일이나 임베디드 상에서 지속적으로 실행되기 위한 알고리즘의 효율성이다. 본 논문에서는 정확도보다 효율성에 Focusing하여 구조 설계를 하였으며 대부분의 실험에서 사용된 모델은 Inference Time의 연산량이 1.5 billion multiply-adds를 유지하도록 설계하였다.

<br>

본 논문에서 제안한 구조인 Inception은 "Network In Network" 논문과 “We Need To Go Deeper”라는 인터넷 밈에서 유래한 이름이다. 이때 "Deep"은 아래와 같은 두 가지 의미를 갖는다.

```
1. Inception Module의 도입

2. Network의 깊이 증가
```

Inception은 "Network In Network"의 논리로부터 영감을 받고, "Provable Bounds for Learning Some Deep Representations"에서 진행된 이론적 연구가 지침이 되었다. Inception 구조의 이점은 ILSVRC 2014 분류 및 탐지 분야에서 실험적으로 검증됐으며, 당시의 state of the art보다 뛰어난 성능을 보였다.

<br>

## Motivation and High Level Considerations

Deep Neural Networks에서 성능을 향상 시키는 가장 단순한 방법은 네트워크의 사이즈를 키우는 것이다. 하지만 이는 아래와 같은 두 가지 문제점이 존재한다.

```
1. 사이즈가 큰 네트워크는 전형적으로 많은 수의 파라미터를 의미하며, 많은 수의 파라미터의 존재로 과적합이 발생하기 쉽다. 
특히, 학습 데이터 셋의 Label의 개수가 제한되어있다면 더욱이다. 

2. 균일하게 증가한 네트워크의 사이즈는 컴퓨터 자원 소비량을 급격하게 증가시킨다.
```

<br>

위와 같은 문제점들을 해결하기 위한 근본적인 방법을 Fully Connected Layer의 구조를 Sparsely Connected Architectures로 변환하는 것이다. "Provable Bounds for Learning Some Deep Representations"에 따르면 거대한 Sparse Deep Neural Network으로 데이터 셋의 확률 분포가 표현 가능하다면, 최적의 네트워크는 마지막 Layer의 Activation의 Correlation Statistics를 분석하고 상관 관계가 큰 뉴런들끼리 묶으면서 구성할 수 있다. 수학적으로 엄격한 상황은 아니지만, Hebbian Principle과 일치하는 것을 통해 실제 상황에서도 적용할 수 있다는 사실을 얻을 수 있다.


<br>

## Architectural Details

