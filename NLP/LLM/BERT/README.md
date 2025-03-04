# [BERT : Pre-training of Deep Bidirectional Transformers for Lanuguage Understanding](https://aclanthology.org/N19-1423.pdf) (NACCL 2019)

<br>

## Abstract

<br>

본 논문에서는 Bidirectional(양방향) Encoder Representations form Transformers(BERT) 모델을 제안한다.

해당 모델은 레이블링 되지 않은 데이터로 사전 학습을 해 하나의 ouput layer만 추가한 fine-tuning을 통해 여러 task(i.e. QA, Language Inference)에서 SOTA를 달성했다.

<br>

## Introduction

<br>

기존의 undirectional 구조의 모델들은 이전의 토큰들만 참고할 수 없다는 한계가 존재한다.

이러한 한계는 sentence-level task에서 최적의 결과를 낼 수 없을 뿐더러, QA와 같은 token-level task에서는 양방향 문맥을 파악하지 못하는 것은 매우 치명적이다.

본 논문에서 제안하는 BERT는 'Masked Language Model(MLM)'을 사용함으로써 해당 문제를 해결한다.

MLM은 input으로 들어가는 몇몇의 tokens에 랜덤하게 마스킹을 하고 context 내에서 마스킹된 토큰의 original vocabulary id를 예측하는 것이다.

사전 학습된 Left-to-Right LM과 달리, MLM은 left, right 문맥을 모두 융합한 representation 생산이 가능케 한다.

추가적으로 text-pair representations이 사전 학습된 MLM은 'Next Sentenece Prediction' task에도 쓰인다.

본 논문이 기여하는 내용은 아래와 같다.

```
1.  Language Representation에서 양방향 사전학습의 중요도를 입증한다.
    양방향 사전학습을 위해 BERT는 MLM을 사용했다.

2.  사전학습된 Representations은 heavily-engineered task-specific architectures에서 
    요구되는 것을 많이 줄인다.

3. BERT는 11개의 NLP tasks에서 SOTA를 달성했다.
```

<br>

## Related Work

<br>

ELMo는 기존의 left-to-right LM만이 아닌 right-to-left LM도 사용해 두 모델의 contextual representation을 concat한다.

ELMo와 task-specific architure를 함께 사용하여 여러 NLP tasks에서 SOTA를 달성했다.

레이블링 되지 않은 데이터로 사전 학습하고 task에 맞게 fine-tuning하는 접근 방식은 scratch부터 학습하는 데에 적은 양의 파라미터만을 요구한다.

이러한 장점을 OpenAI의 GPT 차용했고, 많은 sentence-level tasks에서 SOTA를 달성했다.

<br>

## BERT

<br>

본 연구에서 제안하는 framework는 _pre-training_ , _fine-tuning_ 두 단계로 나눌 수 있다.

Pre-training 과정에서는 unlabeled data로 모델 학습을 진행한다.

Fine-tuning 과정에서는 pre-training 과정에서 학습된 가중치를 지닌 BERT가 downstream task의 labeled data를 이용해 full parameter-tuning을 진행한다.

동일한 pre-trained Model로부터 각 task에 맞게 fine-tuning 하는 구조이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/8af3c0e7-4784-43c2-8f46-b5f0cc4d310e" width='700' height='400'>

</p> 

<br>

BERT의 특별한 점은 Figure를 보면 알다시피 서로 다른 tasks라도 output layer를 제외하고는 BERT 자체의 구조는 변하지 않는다는 것이다.

<br>

### Model Architecture

<br>

BERT 모델은 Multi-layer bidirectional Transformer encoder 기반이다.

```
L : Number of layers
H : Hidden Size
A : Number of self-attention heads
```

본 연구에서는 사이즈에 차이를 둬 두 개의 BERT를 제안하는데, 두 BERT 각각 다음과 같다. 

BERT_BASE(L=12, H=768, A=12 | Total Parameter=110M), BERT_LARGE(L=24, H=1024, A=16 | Total Parameter=340M)

BERT_BASE는 OpenAI GPT와 모델 크기가 같은데, GPT는 Undirectional Transformer를 사용했지만 BERT는 Bidirectional Transformer를 사용했다는 차이점이 있다.

<br>

### Input/Output Representations

<br>

BERT를 fine-tuning해서 다양한 task에서 사용하려면, input representation이 모호하지 않은 representation이 들어가야 한다.

학습하는 시 문장을 기존의 방법과 다르게 사용한다.

1. 문장을 분리하는데 special token인 < sep >을 사용한다.
2. 모든 토큰을 나타낼 수 있는 학습된 embedding을 사용한다.

<br>

### Pre-training BERT

<br>

BERT는 pre-training시 left-to-right model이나 right-to-left model을 사용하지 않고 아래에서 설명할 두 개의 비지도 task를 사용한다.

#### Task 1 : Masked LM

Bidiretional Representation을 위해 Masked 확률을 설정해 랜덤으로 Masked된 tokens을 예측한다.

이러한 구조를 MLM이라고 부른다.

Final hidden layer에서 softmax를 통해 Masked token을 vocabulary에서 어떤 word인지 찾는다.

본 실험에서는 각 sequence의 단어가 masked 될 확률을 15%로 설정했다고 한다.

해당 방법은 bidirectional pre-trained model을 얻을 수 있지만, pre-traininig과 fine-tuning이 mismatch하는 결과를 낳는다는 문제가 있다.

그 이유는, fine-tuning 시 [MASK] 토큰이 나타나지 않기 때문이다.

이러한 문제를 해결하고자 15% 확률로 masked된 token을 전부 [MASK] 토큰으로 변환하는 것이 아니라, 80%는 [MASK]로 변환하고 10%는 무작위 단어로 변환한다.

<br>

#### Task 2 : Next Setence Prediction (NSP)

Question Answering(QA), Natural Language Inference(NLI)와 같은 중요한 downstream task는 두 문장 사이의 relationship을 이해하는 것을 기반으로 한다.

따라서 모델을 pre-training할 때, 간단하게라도  _next sentence prediction_ task를 수행할 정도가 되어야 한다.

이러한 방법은 QA와 NLI에서 매우 효과적인 것을 확인할 수 있었다.

따라서, BERT는 pre-training 시 sequence 학습도 진행하였다.

<br>

#### Pre-training Data

Pre-training할 때, BooksCorpus(800M words) + English Wikipedia(2,500 words)을 이용해 단어와 sequence를 학습했다.

<br>

### Fine-tuning Bert

<br>

BERT는 Transformer의 self-attention mechanism으로 인해 많은 downstream tasks를 해낼 수 있었다.

BERT를 전체 파라미터 fine-tuning을 진행했는데, 각 task에 맞게 input과 output을 간단하게 바꿔 끼기만 하면 됐다.

<br>

## Experiments

<br>

각 task에 맞게 fine-tuning한 BERT를 이용해 11개의 NLP Tasks를 진행했다.

```
- GLUE (General Language Understanding Evaluation)
- SQuAD v1.1 (Stanford Question Answering Dataset)
- SquAD v2.0 (SQuAD v1.1에서 짧은 문장 output은 다 빼서 더 복잡한 task)
- SWAG (Situations With Adversarial Generations)
.etc
```

<br>

+alpha (feature-based vs fine-tuning)

```
Feature-Based : 임베딩 Layer는 그대로 두고 그 위의 Layer만 학습하는 방법
Fine-tuning : 임베딩 Layer까지 Update하는 방법
```

<br>

## Conclusion

<br>

BERT는 Good quality의 pre-trained language representation이 가능할 뿐만 아니라, downstream task로의 손쉬운 fine-tuning이 가능하다.

<br>
