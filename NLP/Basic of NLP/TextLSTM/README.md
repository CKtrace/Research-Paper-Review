# Long Short-Term Memory (Neural Computation 1997)

<br>

## Abstract

<br>

Recurrent 기법을 사용해 시간적 정보들을 학습해나간다면 역전파 과정에서의 소요 시간이 크고 특히 Error back flow 문제가 크다.

본 논문에서는 이러한 문제를 LSTM(Long Short-Term Memory)이라는 방법을 제안해 해결하고자 한다.

LSTM은 1,000개를 초과하는 discrete time steps으로 이루어진 시간 차를 학습시킬 수 있다.

또한 해당 방법의 time step마다와 weight의 computational complexity는 O(1)이다.

여러 실험에서 LSTM은 비교 방법들보다 성능적, 시간적 측면에서 우수했다.

마지막으로 LSTM은 RNN으로 해결하지 못한 복잡한 artificial long time lag tasks를 해결했다.


<br>

## Introduction

<br>

기존의 RNN은 시간에 따라 정보를 잊어버리는 문제가 존재하는데, 이는 여러 학습 방법들이 학습이 누적될 때 역전파되는 오차 신호가 Vanishing되거나 Exploiting 되기 때문이다.

LSTM은 Constant Error Carrousels(CEC) 개념을 도입해 일정한 오차의 흐름 유지할 수 있게 한다.

<br>

## Constant Error Backprop

<br>

TBU