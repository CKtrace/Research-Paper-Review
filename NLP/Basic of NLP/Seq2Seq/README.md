# [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215) (NIPS 2014)

<br>

## Abstract

<br>

기존의 Deep Neural Networks (DNNs)은 어려운 task에서 훌륭한 성적들을 거뒀으며, large label을 가진 데이터에서도 좋은 성적을 보여주었다.

하지만 문장과 문장을 mapping 하는데에는 DNN을 사용할 수 없다.

본 논문에서는 end-to-end sequence 학습 방법을 제안하고, 해당 방법은 Multulayered LSTM을 input layer에 그리고 다른 Multilayered LSTM은 decode하는데 사용한다.

기존에 존재하는 phrase-based SMT system보다 BLEU가 높게 나왔으며, LSTM은 긴 문장들에 대해 어려움이 없었다.

또한 LSTM은 단어의 순서에 민감하게 학습되며 상대적으로 능동인지 수동인지에 대한 여부는 중요하게 받아들이지 않았다고 한다.

<br>

## Introduction

<br>

DNN은 flexibility하고 여러 task에서 좋은 성능을 보이지만, 고정된 차원의 크기의 input과 target 벡터만을 적용한다는 문제가 존재한다.

이러한 문제는 미리 길이를 알 수 없는 문장을 표현하기 어렵다는 중대한 문제로 직결된다.

본 논문에서는 해당 문제를 LSTM 구조를 적용해 sequence to sequence 문제를 해결한다.

하나의 LSTM을 이용해 input sequence를 읽어 고정된 큰 차원의 벡터를 얻은 후, 다른 LSTM으로 해당 벡터를 이용해 output sequence를 내놓는다.

Seq2Seq가 WMT'14 English to French translation task에서 BLEU score가 34.81점을 기록했다.

해당 구조는 4개의 LSTM과 간단한 left-to-right Beam search decoder를 사용한 것이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/7e8a502c-3772-4735-ab62-71a7259ebdf5" width='400' height='250'>

</p> 

<br>

LSTM은 매우 긴 Sequence에도 Robust했다고 한다.

또한 재밌는 점은, 긴 Sequence을 거꾸로 넣었을 때 성능이 더욱 좋았고, 이는 많은 단기 종속성을 도입할 수 있기 때문이었다.

단, Sequence를 거꾸로 넣었다는 것은 train과 test set에서의 target sentence가 아닌 source sentenece만의 단어 순서를 뒤집었다는 것이다.

추가로 SGD는 LSTM이 긴 Sequence도 문제없이 학습할 수 있게 해주는 것도 발견하였다.

LSTM의 유용한 역할은 바로 길이의 제한이 없는 input sequence를 받아 고정된 차원 벡터로 나타내는 것이다. 

<br>

## The model

<br>

RNN은 sequence to sequence를 쉽게 매핑할 수 있는 방법이다.

하지만, RNN은 input과 output의 길이가 다른 문제에 적용하기 어렵다는 단점이 존재한다.

가장 간단한 전략은 하나의 RNN을 사용해 input sequence를 고정 크기 벡터에 매핑하고 다른 RNN으로 해당 벡터를 target sequence에 매핑하는 것이다.

이때, 장기 종속성으로 인해 RNN을 대신해 LSTM을 사용하는 것이 이 세팅을 성공적으로 해내는 방법이었다고 한다.

본 연구에서는 각 문장의 끝을 알리는 < EOS > 기호를 적용해 모델이 가능한 모든 길이의 Sequence에 대한 분포를 정의할 수 있게 하였다.

첫 번째로, 서로 다른 두 개의 LSTM을 사용한 것은 두 LSTM을 사용해도 Computational Cost가 무시해도 될 만큼만 증가하고 여러 언어를 동시에 학습할 수 있다.

두 번째로, deep LSTM은 shallow LSTM보다 성능이 좋은 것을 발견했고, 본 논문에서 제안하는 모델은 4-layer deep LSTM을 사용하였다. 

세 번째로, input sentence를 역순으로 넣는 것이 매우 가치 있는 것임을 발견했다.

예를 들어 a, b, c sequence와 d, e, f sequence가 있다면, 이를 c, b, a sequence와 d, e, f sequence로 매핑하는 것이다.

이때, d, e, f는 a, b, c의 translation이다.

해당 방법은 a와 d를 가깝게 해, SGD가 더욱 쉽게 input과 output 사이의 'establish communication'을 가능케 한다.

<br>

## Experiments

<br>

해당 연구에서는 WMT'14 English to French 데이터셋을 사용했는데, 제안한 모델은 348M의 French words와 304M English words로 이루어진 12M개의 문장을 학습했다.

가장 빈도가 높은 단어 중 16만 개를 source language, 8만 개를 target language에 사용하였고 어휘에 없는 단어는 UNK 토큰으로 대체하였다.

```
추가적인 모델 학습 접근 방법은 Introduction에서 설명한 내용과 많이 겹쳐 Skip
```

데이터셋의 모든 문자의 길이가 달라 최대 길이 문장 길이에 미니 배치 사이즈를 맞추면 비교적 짧은 문장들로 인해 자원 낭비가 심해졌다.

해당 문자의 해결 방법은 각 batch마다 비슷한 length를 가진 sequence들로 이루어지게끔 normalization을 진행했다고 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/6e92aad4-f0bc-4a13-9c23-b537b36043b9" width='500' height='250'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/00c8e013-e328-413a-bcb0-305db85938e1" width='700' height='250'>

</p> 

<br>

OOV words가 존재함에도 불구하고 기존의 방식들보다 좋은 성적들을 보여주었다.

실험에서 LSTM은 길이가 긴 sequence 또한 잘 소화해낸다는 것을 발견했고 하단의 table과 figure가 이를 보여준다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/8eddd50d-546e-417e-bc2c-55f401bb0afc" width='700' height='400'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/1a209238-40d8-4dae-8571-d93edc533a8c" width='500' height='400'>

</p> 

<br>

## Conclusion

<br>

본 연구에서는 source sentences의 word를 역순으로 넣었을 때 성적이 더욱 좋았던 부분과 LSTM의 긴 문장을 잘 처리해내는 능력이 놀라웠다고 한다.

또한 RNN을 사용했을 때, input sequence 길이와 output sequence 길이가 같은 문제를 해결했다는 것에도 의의가 있다.

<br>
