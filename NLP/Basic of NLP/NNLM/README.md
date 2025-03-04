# [A Neural Probabilistic Language Model (JLMR 2003)](https://papers.nips.cc/paper_files/paper/2000/file/728f206c2a01bf572b5940d7d9a8fa4c-Paper.pdf)

<br>

## Abstract

<br>

통계적 언어 모델의 목표는 문장 속 단어들의 순서의 Joint probability function을 학습하는 것이지만, 이는 '차원의 저주'로 인해 본질적으로 어렵다는 문제가 있다.

전통적인 N-gram 기법으로 해당 문제에 대해 효과적이었지만, 너무 짧은 overlapping Sequence를 이용한다는 문제점이 있었다.

본 논문에서는 '차원의 저주'를 단어들의 분산 표현을 학습하는 방법으로 해결하는 방법을 제안한다.

제안하는 모델은 각 단어에 대한 분산 표현과 단어 시퀀스에 대한 확률을 구한다.

<br>

## Introduction

<br>

문장에서 가까운 위치에 있는 단어들이 통계적으로 더 종속적이라는 사실을 활용해 통계적 모델링 문제의 어려움을 줄일 수 있고 이를 활용한 것이 N-gram이다.

만약, training corpus에서 보지 못한 새로운 n words의 조합이 나타나면 예측하기 어려워진다.

N-gram은 1~2개 단어 정도만 사용하고 그것보다 더 멀리 있는 단어는 사용하지 않으며, 유사도를 고려하지 않는다.

예를 들어, 'The Cat is walking in the bedroom'이나 'A Dog is running in a room'이 비슷한 문장이지만, N-gram 모델로는 Semantic / Grammatical한 유사도를 잡아내지 못한다.

<br>

### Fighting the Curse of Dimensionality with Distributed Representations

해당 파트를 요약하자면, 초록에서 설명한 것처럼 Vocabulary에 있는 단어들을 word feature vector로 표현하고 이들을 joint probability function으로 표현하고 최종적으로 word feature vector와 probability function의 parameter를 동시에 학습하는 것이다.

Feature Vector는 각 단어들이 Vector 공간에 놓일 수 있게 하는데 이는 단어들 간의 유사도를 구할 수 있게 해준다.

이렇게 표현된 Feature들은 Vocabulary의 사이즈보다 훨씬 작다.

유사한 단어들은 유사한 Feature vector를 가질 것이고 probability function 또한 이들의 feature value들에 대해서도 smooth할 것이기에,
Feature의 작은 변화는 probability에도 작은 변화를 부과할 것이다.


<br>


### Relation to Previous Work

고차원의 이산 분포를 모델링하기 위해 신경망을 쓴다는 아이디어는 이미 Joint probability에 유용하다는 것은 이전 논문들에서 보여졌으며,
이것은 Language Modeling까지 발전하였다.

하지만 본 논문에서는, 각 문장에서의 각 단어들의 역할이 아닌 각 단어 시퀀스들의 분포에 대한 통계적 모델을 학습하는데 집중한다.

본 연구에서는 단어들 간의 유사도를 표현하기 위해 word feature vector를 사용한다고 한다.


<br>

## A Neural Model

요약하자면, N-gram을 본질로 삼되, one-hot encoding은 Large Vocabulary에 적합하지 않다는 문제점을 해결하기 위해 Distributed Representation을 이용한다.

본 논문에서 제안하는 모델은 임베딩 기법으로, n-1개 단어 순서 기반으로 n번 째에 등장할 단어를 맞추는 모델이다.

<br>

![Image](https://github.com/user-attachments/assets/99e00a2a-e7bc-4dde-848e-e910ccce6b9f)

<br>

```

t = 예측할 단어의 위치
n = 입력되는 단어의 개수


|Input Layer|

index for ~ 는 t-(n-1)부터 t-1까지 단어 벡터 위치를 나타내는 One-hot vector와 랜덤한 초기값을 갖는 C 행렬이 내적 되어
Word Feature Vector가 나온다.

이렇게 나온 값들을 Concatenate하여 일자로 쭉 세운다.



|Hidden Layer|

tanh 함수를 이용해 Score Vector를 구한다.

y = b + Wx + U*tanh(d+Hx) (b, W, U, d, H는 매개변수)

(b & d : 각각 bias term  H : Hidden layer's weights W : input layer & output layer에 direct connection을 만들 경우의 weights)



|Output Layer|

y 값에 Softmax를 적용시켜 정답 one-hot vector와 값 비교 후 역전파를 통해 학습한다.

```

<br>

## Significance of Paper

<br>

본 모델의 아쉬운 점은 학습하면서 Update 해 나가야 할 Parameter가 C, H, U 등이 있어 계산복잡성이 높다는 점이다.

하지만, 해당 연구를 통해 단어를 Vocabulary의 크기보다 작은 차원의 벡터로 표현하는 시도를 통해 Word2Vec가 탄생할 수 있었으며, 
Word2Vec는 위의 문제를 고치기 위해 학습해야 하는 Parameter를 줄여냈다고 한다.

<br>