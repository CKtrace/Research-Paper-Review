# Improving Language Understanding by Generative Pre-Training

<br>

## Abstract

<br>

Natural Language Understanding은 넓은 범위의 다양한 tasks들로 이루어져 있다. : textual entailment, QA, semantic similarity assessment, document classification

Unlabeled data는 많고, labeled data는 적은 경우 적절한 성능을 보이는 model을 학습 시키는 것은 도전적이다.

이전 연구들과 달리, 해당 연구에서는 model 구조의 적은 변화로 효율적인 fine-tuning 하는 방법을 소개하고자 한다.

본 논문에서 제안한 GPT-1의 wide range benchmarks에서 얼마나 효과적인지 입증하고, task에 맞게 fine-tuning 했을 때는 12개의 task 중 9개 부문에서 SOTA를 달성했다.

<br>

## Introduction

<br>

Raw text으로 효과적으로 학습하는 능력은 NLP 분야에서 지도 학습의 종속성을 완화하는데 필수적이다.

대부분의 Deep Learning 방법은 labeled data를 요구하며, 이러한 점은 annotated resource가 부족한 분야에서의 적용 가능성이 제한된다.

Labeled data가 많더라도, unlabeled data로 quality 높은 representation을 배워놓는다면, labeled 데이터로 학습시 중요한 부스트 역할을 한다고 한다.

즉, unlabeled data로 pre-training을 진행하고 labeled data로 fine-tuning을 하면 좋은 성능을 얻을 수 있다는 것이다.

본 연구에서의 Setup은 two-stage로 나뉜다.

1. Unlabed Data로 Neural Network Model의 초기 파라미터를 학습한다.
2. 각 task에 맞게 학습한 파라미터를 시작으로 Labeled data를 이용해 학습한다.

Language Understanding Task에서 네 가지 types의 접근을 했다.

1. Natural Language Inference
2. QA
3. Semantic similarity
4. Text Classification

<br>

## Related Work

<br>

### Semi-supervised learning for NLP

<br>

과거의 연구들에서는 unlabeled corpora로 학습 시킨 word embedding을 사용하였을 때 얻는 이점들에 대해 입증하였다.

그러나 이러한 접근은 word-level information로의 변환이 주이고 해당 연구에서는 higher-level semantics을 얻는데 초점을 둔다.

최근의 접근들은 unlabeled data에서 word-level semantics 이상의 것을 학습하고 이용하고자 하는 탐구들이 이루어졌다.

Unlabeled corpus를 학습시켜 얻는 Phrase-level or sentence-level embedding은 다양한 target tasks에 적절한 vector representation으로 encoding하는데 쓰인다.

<br>

### Unsupervised pre-training

<br>

Unsupervised pre-training은 Semi-supervised Learning의 special case이며 good initialization point를 찾는 것이 목표이다.

최근 연구들에서 Unsupervised pre-training은 image classification, speech recognition 등 다양한 task에서 deep neural networks이 학습을 도우는데 쓰인다.

해당 연구로부터 가장 최근의 연구에서는 pre-training된 Neural Network을 supervised로 target task에 맞게 fine-tuning하여 text classification에서 성능 개선을 했다.

하지만 해당 연구에서 pre-training을 통해 linguistic information을 잡아내는데에는 성공적이었지만, LSTM을 사용해 short range의 예측 밖에 할 수 없었다.

이러한 문제점 때문에, 본 연구에서는 LSTM 대신 Transformer를 채택했고, 이는 linguistic structure에서 longer-range를 잡아낼 수 있었다.

또한 본 논문에서 제안한 GPT-1은 wider range of task에서 효과적인 모습을 보인 것 또한 증명했다.

<br>

## Auxiliary training Objectives

<br>

Auxiliary Unsupervised training Objectives은 semi-supervised의 대안이다.

기존 연구에서는 Auxiliary language modeling objective 그들의 target task에 추가해 sequence labeling tasks에서 성능 개선을 이끌어냈다.

본 연구에서도 Auxiliary objective을 사용하지만, target tasks에 해당하는 몇몇의 linguistic aspects을 미리 학습이 진행됐다.

<br>

## Framework

<br>

해당 연구의 학습 절차는 둘로 나눌 수 있다.

1. Learning a high-capacity language model on a large corpus of text
2. Target Task의 Labeled data로 학습된 파라미터를 활용해 Fine-tuning

<br>

### Unsupervised pre-training

Unlabeld corupus의 토큰 집합 U= {u_1, u_2, ..., u_n}은 아래의 likelihood을 최대화 시키는 것이 목표이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/475ac8ad-6c54-4962-b1f9-8e5bce51d0ec" width='350' height='70'>

</p> 

<br>

k는 context window size이며, 조건부 확률 P는 theta parameter를 사용한다.

해당 파라미터들은 Stochastic Gradient Descent를 이용하여 학습된다.

본 실험에서는 Multi-Layer Transformer decoder를 Language model로 채택했다.

<br>

전체적인 수식은 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/f6197338-ac4a-4820-a189-a2efc936505d" width='450' height='200'>

</p> 

<br>

### Supervised fine-tuning

<br>

Labeled Data C와 sequence of input tokens x^1,...,x^m와 label y로 방정식이 이루어져있다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/c2dd316a-f132-4c0a-9803-b4f54f1a799a" width='350' height='70'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/57491631-b400-4a3a-81cb-e1233859a0ca" width='350' height='70'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/5780b011-d05a-4808-902c-ad6caeefcb98" width='350' height='70'>

</p> 

<br>

이전에 언급했던 GPT-1의 4가지 task를 적용하면 각각의 구조는 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/dd2f2cb3-fd7c-48de-a71e-c66073f9c1dc" width='700' height='500'>

</p> 

<br>

## Experiments

<br>

### Model specification

<br>

12-layer decoder-only transformer with masked self-attention heads (768 dimenstional states + 12 attention heads)

Position-wise feed-forward networks는 3072 차원의 inner states를 갖는다.

```
- Adam with Max LR == 2.5e-4
- BPE
- L2 regularization w == 0.01
- GELU (activation function)
```

### Fine-tuning details

<br>

```
Dropout == 0.1
LR == 6.25e-5
Batchsize == 32
Warmup == Over 0.2% of training
Lambda(Weight) == 0.5
```

4개의 task 중 12개의 dataset을 사용했고 해당 dataset은 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/956e06b6-0f10-400d-810b-85c369007a07" width='700' height='250'>

</p> 

<br>

아래의 table은 12개 중 9개의 SOTA를 달성한 것을 보여주는 테이블이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/7d6d8bd9-d587-4a58-a2a1-4a84d971e782" width='700' height='400'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a5cffd85-7018-48be-adcb-21b65a7b3bd2" width='700' height='300'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/7c4d947c-a129-41c1-8e2b-f51ab65f06fc" width='700' height='300'>

</p> 

<br>


## Conclusion

<br>

GPT-1은 기존의 방식들과 달리 각 task에 맞게 architecture를 설계하는 것이 아닌, Generative pre-training모델과 discriminative fine-tuning 모델을 제안했다.

다양하고 길이가 긴 문장을 pre-train 해서 Long-range dependency를 해결하고 12개 중 9개의 task에서 SOTA를 달성했다.

마지막으로 LSTM을 썼을 때보다 Transformer를 썼을 때의 성능이 월등히 높아지는 것을 보여주며, Transformer의 중요성을 다시 한 번 보였다.

(이후 대부분의 Generative Model은 Transformer를 사용한다고 봐도 무방할 정도로 많이 사용한다.)

<br>
