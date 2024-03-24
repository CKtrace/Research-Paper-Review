# [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

<br>

## Abstract

<br>

본 연구에서는 적절한 Convolution 분해와 과감한 Regularization을 통해 추가된 연산량을 효율적으로 사용하는 것에 초점을 둔 Network 크기 증가 방법에 대해 탐구한다.


<br>
<br>

## Introduction

<br>

분류 부분에서 성능이 높아질수록 다른 분야에 응용했을 때도 성능이 높아진다는 사실을 발견했다.

이는 개선된 Convolution 아키텍처는 아키텍처 자체의 성능이 발전하면 컴퓨터 비젼의 대부분의 분야, 즉 고품질의 학습된 시각적 특징에 의존하는 분야들에서도 성능이 발전한다는 것이다.

또한 네트워크의 발전에 따라 기존의 AlexNet이 직접 손으로 짠 방법보다 더 좋지 못했던 분야에서도 Convolution 네트워크가 사용될 수 있게 되었으며 그 예가 검출에서의 제안 생성이라고 한다.

Inception의 연산량은 VGGNet 또는 그보다 더 높은 성능을 갖는 모델들보다 훨씬 적다.

이러한 점은 Inception을 한정된 메모리 혹은 계산량이 내포된 상황에서 상당한 양의 데이터를 처리해야 하는 빅데이터 상황에서의 사용을 쉽게 할 수 있게 해준다.

허나 Inception 아키텍처의 복잡한 구조는 Network를 변경시키는데 어려움을 겪게 한다.

만약 아키텍처가 단순하게 커진다면, 계산에서 얻은 이점들을 사라지게 할 것이라 한다.

<br>
<br>

## General Design Principles

<br>

본 파트에서 설명하는 원칙들을 벗어나면 네트워크 성능은 떨어지고, 원칙들을 충족하면 아키텍처가 일반적으로 개선되는 경향을 보인다고 한다.