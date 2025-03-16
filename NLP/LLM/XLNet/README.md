# [XLNet: Generalized Autogressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237) (NIPS 2019)

<br>

## Abstract

<br>

BERT는 masked positions로부터 의존적이지 못하고, Mask 토큰은 pre-training에만 사용되며 fine-tuning에는 사용되지 않아 pretrain-finetune 모순을 겪는다.

Generalized autogressive pretraining 모델인 XLNet을 제안하며, 이는 두 가지 Method를 지니고 있다.

1. Bidirectional contexts를 학습할 수 있다.
2. Autoregressive formulation을 통해 BERT의 한계를 극복한다.

추가적으로 XLNet은 Transformer-XL을 채택해 Autoregressive Model SOTA를 달성했다.

XLNet은 BERT를 20가지의 task에서 더욱 개선된 성능을 보였다.

<br>

## Introduction

<br>

NLP에서 unsupervised representation learning이 매우 성공적이었고, 해당 방법은 대규모 unlabeled text corpora로 학습한 첫 pretrain neural networks이고 이를 finetune 모델을 하거나 downstream tasks로의 representations을 한다.

다른 Unsupervised pretraining objectives을 찾고자 하는 노력이 있었고, 그 중 AR(Autoregressive) Language Modeling과 AE(Autoencoding)이 매우 성공적이었다.

기존의 AR Language Model(i.e. GPT)은 uni-directional context를 encoding하기 위해서만 학습되었기에, deep bidirectional contexts modeling에 효과적이지 못했고, 이는 bidirectional context information을 주로 요구하는 downstream language understanding tasks에서의 좋지 못한 성능을 가져온다.

반면에 AE based Model(i.e. BERT)는 density한 estimation이 불가하지만, 손상된 input을 복원하는데에 초점을 맞춘다.

해당 모델 중 대표적으로 BERT가 있는데, 주어진 input token sequence에 일정 확률로 기존의 token 대신 [MASK]로 대체하고 이를 복원하기 위한 학습을 진행한다.

Density estimation이 불가하다는 단점 때문에 BERT는 Reconstruction에서 bidirectional contexts를 사용했다.

Bidirectional Information을 추가하는 방법은 AR language model과의 gap을 줄일 수 있었다.

그러나 pretraining 때 사용한 [MASK]는 finetuning할 때는 사용하지 않아 pretrain과 finetune간의 모순이 존재한다.

또한 BERT는 token을 예측할 때, 주어진 각각의 unmasked tokens과 독립적으로 진행한다.

따라서, 본 논문에서는 AR과 AE의 장점들을 담은 XLNet을 제안한다.

XLNet은 많은 task에서 BERT보다 뛰어난 성능들을 보여주었다.

<br>

## Proposed Method

<br>

### Background

<br>

AR language modeling에서는 forward autoregressive factorization에서의 likelihood를 최대화하는 것이 목표이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/75e652bb-a5e8-4fa5-94e5-e2b0f46cb936" width='500' height='100'>

</p> 

<br>

Denoising AE 기반인 BERT는 text sequence x에 masked한 token x^를 정답 x^-으로 reconstruct하는 것이 목표이며 수식은 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/24fe4589-8fef-4c63-9827-adf6c67fb64a" width='500' height='100'>

</p> 

<br>

### Objective: Permutation Language Modeling

<br>

AR language modeling과 BERT는 서로 상반되는 장점을 가지고 있고, 이를 통합하면 두 방식의 단점을 상쇄하고 각각의 장점을 얻을 수 있다.

AR과 AE을 통합한 XLNet의 목표는 아래의 수식을 최대화하는 것이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/24a5985a-40ee-41f4-b33e-29aa2730e565" width='500' height='100'>

</p> 

<br>
