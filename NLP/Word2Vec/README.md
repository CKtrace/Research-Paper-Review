# [Efficient Estimation of Word Representations in Vectore Space (ICLR 2013)](https://arxiv.org/pdf/1301.3781)

<br>

## Abstract

<br>

본 논문에서는 매우 큰 데이터 셋에서의 Continuous vector representations of words를 계산하는 두 가지 모델 구조를 제안한다.

단어 유사도 부분에서의 성능 이전에 소개된 여러 Neural Network 기반의 방법들보다 좋은 성능을 보인다.

해당 연구에서는 적은 Computational Cost와 더불어 정확도 측면에서의 상당한 개선을 보여준다.
(i.e. 1.6B(16억)개의 단어로 구성된 데이터셋에서의 high quaility word vector들을 학습하는데 하루가 덜 걸린다.)

이러한 word vector들은 Syntactic(문장 내에 있는 구성 요소의 순서) & Semantics(구성 요소의 의미) 단어 유사도를 측정하기 위한 테스트 셋에 State-of-the-art Performance를 보인다.

<br>

## Introduction

<br>

해당 논문에서 저자들은 새로운 모델 구조를 발전시켜 word vector가 Linear Regularities among words를 보존하면서 정확도를 최대화 시키는 것을 시도했다.

<br>

```
i.e. Lienear Regularities Words

vector("King") - vector("Man") + vector("Woman") = vector("Queen")
```

<br>


과거의 연구에서는 continuous vector로 표현된 단어들에 대한 연구가 꾸준히 있었고 그 중 NNLM을 기반으로 한 연구들이 꾸준히 이어졌다고 한다.

하지만, 연구가 거듭되면서 결과와 반대로 Computational Cost는 기하급수적으로 증가하게 되었다.

<br>

## Model Architectures

<br>

해당 연구에서는 단어들의 분산 표현을 Neural Network로 학습하는 데 초점을 두었고, 이미 해당 방법은 LSA(Latent Semantic Analysis)보다 Linear Regularities를 잘 보존한다는 것은 잘 알려져 있다.

먼저 타 모델들과 비교하기 위해 모델을 Fully train 하는데 필요한 파라미터 수, 즉 모델의 Computational Complexity를 정의하였다.

<br>

```
Training Computational Complexity는 O에 비례한다.


O = E x T x Q


E : Epoch (Common choice -> 3 ~ 50)
T : Number of the words in the training set (Up to one billion(10억까지))
Q : 제안하는 각 모델에서 정의
```

<br>

두 번째로 Computational complexity를 줄이며 정확도는 최대화를 시키는 시도를 했다.

<br>

### Feeforward Neural Net Language Model (NNLM)

<br>

NNLM Computational Complexity는 아래와 같다.

```
Q = N x D + N x D x H + H x V

Q : Computational Complexity
N : N words
V : Vocab Size
D : Input Layer
H : Hidden Layer
```


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/4fc90341-97d2-4a5a-8e56-eb84a2733fab" width='400' height='700'>

</p> 

<br>

H x V 부분에서의 연산량은 후속 연구에서 Hierarchical Softmax로 연산 해야 될 개수가 V개 였는데 이를 log_2(V)로 줄일 수 있었다고 한다.

Hierarchical Softmax는 기존의 Softmax에 근사한 값을 Sigmoid를 이용해 근사 시키는 방법이다. 

해당 논문에서 제안하는 Skip-gram에서도 Huffman Tree를 이용한 Hierarchical Softmax를 사용해 연산량을 줄인다.

설명을 덧붙이자면, Word2Vec에서 Vocabulary에 있는 모든 단어를 Huffman Tree들의 Leaf로 갖게 한다.

또한 단어의 등장 빈도에 따라 깊이를 달리 하는데, 자주 등장하는 Frequent Word로 깊이가 얕은 곳에 위치하고 가끔 등장하는 Rare Word는 깊게 배치한다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/6a3c850c-82b8-4faf-9aab-56a1d3758935" width='500' height='500'>
    Image Source Link : https://uponthesky.tistory.com/15
</p> 

<br>



Huffman Tree를 이용한 Hirarchical Softmax의 연산량은 log_2(Unigram_perplexity(V))이고, 기존의 Balanced Binary Tree를 사용한 Hierarchical Softmax의 연산량은 log_2(V)이다.

이렇게 N x D 에서의 Computational Complexity의 문제를 해결할 수 있었지만, N x D X H 에서의 Computational Complexity의 문제는 해결하기 어렵기에 해당 연구에서는 hidden layer를 갖지 않고 Hierarchical Softmax에 크게 의존하는 모델 구조를 선보이고자 한다.


<br>

### Recurrent Nueral Net Language Model (RNNLM)

<br>

NNLM은 직접 Context Length(N)를 지정해줘야 한다는 단점이 있다.

RNNLM은 이러한 단점을 해결하고 더 복잡한 패턴을 작은 Nueral Networks로 효율적으로 표현하는 RNN을 사용한다.

RNN은 NNLM의 구조와 달리 projection layer가 존재하지 않고, layer 구성은 input / hidden / output layer로 구성되어 있다.

또한 RNN의 특이한 구조 중 하나인 Time Delayed Connection은 short term memory를 갖게 해 과거의 정보와 현재의 정보를 결합하여 학습할 수 있게 해준다.

RNN의 Computational Complexity는 아래와 같다.

<br>

```
Q = H x H (= D x H) + H x V
```

<br>

Word Representation의 크기인 D는 hidden layer의 크기인 H와 같으므로 H x H로 표현한다.

또한 H x V는 위에서 설명한 것과 같이, hierarchical softmax를 이용해 H x log_2(V)로 줄일 수 있다.

그러므로, 가장 큰 Complexity는 H x H 부분에서 온다.

<br>

## New Log-linear Models

<br>

해당 섹션에서는 Computational Complexity를 최소화하는 두 가지 모델을 제안한다.

이전 섹션에서 보았듯이, Nueral Network의 hidden layer에서의 연산량이 문제였는데, 이를 피하고 학습의 효율성을 극대화 하는 방법을 고안하고자 했다.

<br>

### Continuous Bag-of-Words Model (CBOW)

<br>

CBOW는 NNLM과 유사하지만, non-linear hidden layer를 없애고 projection layer는 모든 단어들과 공유하는 방법을 사용한다.

Projection layer를 모든 단어들과 공유한다는 것은 word vector에 평균 값이 들어가게 되는데 이러한 구조를 Bag-of-Words(BOW)라고 하며 이전에 projection 된 단어들은 영향을 주지 않는다.

CBOW는 log-linear classifier를 4개의 과거 단어와 4개의 미래 단어를 input으로 사용해 가운데에 위치한 현재 단어를 훈련하여 좋은 성능을 보인다.

CBOW의 Training Complexity는 아래와 같다.

<br>

```
Q = N x D + D x log_2(V)
```

<br>

CBOW는 기존의 BOW와 달리 Continuous distributed representation of the text를 사용한다.

CBOW의 구조에서 주목해야할 부분은 모든 단어들이 동일한 Weight Matrix를 사용해 Input Layer에서 Projection Layer로 Projection 되는 것이다.

CBOW의 구조는 아래와 같다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/6fedbeca-ce8d-4729-a860-debbbea27c3a" width='400' height='500'>
    CBOW Structure
</p> 

<br>



### Continuous Skip-gram Model

<br>

제안한 Skip-gram은 CBOW와 유사하지만, 주변 맥락 단어를 기준으로 현재 단어를 예측하는 것이 아닌 어느 한 단어를 이용해 앞 뒤 단어들을 예측하는 것이다.

Skip-gram의 Computational Complexity는 아래와 같다.

<br>

```
Q = C x (D + D X log_2(V))

C : 단어들 간의 최대 거리
```

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/495613ed-a64b-4f07-b6e2-f5ff64831b73" width='400' height='500'>
    Skip-gram Structure
</p> 

<br>


## Result

<br>

해당 연구에서 제안한 모델을 학습할 경우,
```
Vector(Smallest) = Vector(Biggest) - Vector(Big) + Vector(Small)
```

위와 같은, 관계도 잘 찾아내는 Syntatic task 뿐만 아니라 Semantic task도 잘 수행해낸다.

Word Vector를 훈련시키기 위해 60억 개의 토큰을 포함하는 Google News Corpus를 사용했는데, 가장 빈도수가 높은 100만 개의 토큰으로만 training으로 학습을 진행했다고 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/7c3cd048-49b0-4f82-a90b-9e1a64a45389" width='700' height='200'>

</p> 

<br>

해당 부분에서는 CBOW에서 학습 데이터와 벡터의 차원의 크기와 정확도는 비례하지만 일정 수준 이상부터는 미비한 것을 확인 할 수 있다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/ecfb93f7-825e-4d7d-897e-510992209a9a" width='700' height='300'>

</p> 

<br>


해당 Figure를 통해 RNNLM과 NNLM보다 제안한 두 모델의 성능이 Semantic Task와 Syntactic Task 모두 우수한 것을 확인할 수 있다.

<br>

## Large Scale Parallel Training of Models

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/cd07525e-5a25-42ec-8241-ff9045db9aef" width='800' height='300'>

</p> 

<br>

60억 개의 토큰을 학습하는데 사용했을 때, NNLM은 가장 많은 CPU 코어를 사용했음에도 불구하고 14일이나 걸렸고 학습일이 2일 걸린 CBOW와 비교해 7배나 차이가 난다.

하지만 Skip-gram과 CBOW의 Semantic과 Syntactic 정확도는 NNLM보다 우수하다.

<br>

## Conclusion

<br>

해당 논문에서 제안한 방법이 NLP application에서 중요한 블럭 역할을 할 것이라 기대한다고 하며 마무리한다.