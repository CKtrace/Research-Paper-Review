# [A Neural Probabilistic Language Model (JLMR 2003)](https://papers.nips.cc/paper_files/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf)

## Abstract

<br>

통계적 언어 모델의 목표는 문장 속 단어들의 순서의 Joint probability function을 학습하는 것이지만, 이는 '차원의 저주'로 인해 본질적으로 어렵다는 문제가 있다.

전통적인 N-gram 기법으로 해당 문제에 대해 효과적이었지만, 너무 짧은 overlapping Sequence를 이용한다는 문제점이 있었다.

본 논문에서는 '차원의 저주'를 단어들의 분산 표현을 학습하는 방법으로 해결하는 방법을 제안한다.

제안하는 모델은 각 단어에 대한 분산 표현과 단어 시퀀스에 대한 확률을 구한다.

<br>

## I. Introduction

<br>

문장에서 가까운 위치에 있는 단어들이 통계적으로 더 종속적이라는 사실을 활용해 통계적 모델링 문제의 어려움을 줄일 수 있고 이를 활용한 것이 N-gram이다.

만약, training corpus에서 보지 못한 새로운 n words의 조합이 나타나면 예측하기 어려워진다.

N-gram은 1~2개 단어 정도만 사용하고 그것보다 더 멀리 있는 단어는 사용하지 않으며, 유사도를 고려하지 않는다.

예를 들어, 'The Ccat is walking in the bedroom'이나 'A dog is running in a room'이 비슷한 문장이지만, N-gram 모델로는 Semantic / Grammatical한 유사도를 잡아내지 못한다.

<br>

### I.1 Fighting the Curse of Dimensionality with Distributed Representations

<br>

TBU
