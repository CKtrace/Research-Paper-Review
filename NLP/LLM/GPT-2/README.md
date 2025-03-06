# [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (OpenAI)

<br>

## Abstract

<br>

LM이 다양한 NLP task를 explicit supervision없이 수 백만 개의 webpages인 WebText dataset으로 학습 가능함을 입증했다.

또한 127,000+ 개의 training examples 없이 CoQA에서 55 F1을 달성했고, 이는 4개의 baseline 중 3개를 뛰어넘는 수치이다.

본 논문에서 제안하는 GPT-2는 1.5B parameter Transformer이고, 이는 zero-shot setting으로 Language Modeling Datasets의 8개 중 7개에서 SOTA를 달성했으며, 여전히 WebText에는 underfit 되어있다.

<br>

## Introduction

<br>

현존하는 system들은 특정 분야에 특화되어 있어서 General한 부분이 부족하다.

본 연구의 목표는 다양한 task를 소화할 수 있는 더욱 General system을 만들어내는 것이다.

Single domain datasets으로 training을 진행하는 이러한 현재의 방식이 General system을 만들지 못하는 이유라고 한다.

Multitask learning은 general performance를 개선시키는 것이지만, 현재 NLP는 여전히 초기 상태에 머물러 있다고 한다.

현재 Language Task에서 성능이 제일 좋은 system은 pre-training과 supervised fine-tuning 조합이다.

최근의 연구에서 Transformer와 같은 architecture의 등장으로 LM의 조건부확률을 계산하는 model의 표현력이 향상되었다고 한다.

일반적으로, Single Task를 수행할 때는 P(output|input)으로 확률이 계산되지만, 본 논문에서는 Task도 같이 Condition되어야 한다고 주장하며 P(output|input, task)를 제안했다.

본 논문에서는 LM이 zero-shot setting에서 down-stream tasks에서 우수한 성능을 보이는 접근 방법을 보여주고자 한다.

<br>

## Approach

<br>

### Training Dataset

<br>

본 연구의 접근 방식은 최대한 크고 다양한 dataset으로 학습시켜 다양한 Domain과 Context에서의 가능성을 보여주는 것이다.

본 연구에서 제안한 WebText Dataset은 4천 5백만 개의 link에 관한 text를 담고 있으며 2017년 12월 이후의 Link는 전부 포함하지 않았다.

De-duplication과 heuristic based Cleaning 과정을 거친 최종 Dataset은 8백 만개의 Document, 40GB의 text로 이루어져 있다.

또한 Wikipedia Document는 타 dataset에서도 흔히 보이고, 이 때문에 data overlapping problem 발생 가능성이 있어 Wikipedia Document는 제외했다.

<br>

### Input Representation

<br>

당시 Language Modeling을 위해서는 Lower-casting, tokenization, OOV 처리 등 preprocessing 단계를 거쳐야 하며 해당 preprocessing을 실행하기 위해서는 unicode 문자열을 UTF-8로 변환해 byte-level에서 처리해야 한다.

하지만 현존하는 byte-level LMs은 word-level LMs보다 large scale datasets에서의 성능이 떨어지는 모습을 보인다.

본 연구에서는 standard byte-levle LMs를 large scale datasets인 WebText로 학습시켜 word-level LM과 유사한 performance를 보여준다고 한다.

기존의 BPE(Byte Pair Encoding)은 unicode 문자열이라 Vocabulary Size가 상당히 크지만, 제안한 BBPE를 사용하면 Vocabulary Size를 256으로 줄일 수 있다.

하지만 BBPE는 {dog., dog?, dog!}와 같이 단어의 유의미하지 않은 Variation을 추가하는 단점이 존재하는데, 이럴 경우 Vocabulary Size를 Sub-optimal하게 사용하게 된다.

이러한 단점을 막고자, 문자 수준 이상의 병합을 막는 규제를 추가하였다.

BBPE를 사용함으로써, 어떤 dataset으로 학습해도 LMs이 preprocessing, tokenization, vocab size에 관계 없이 좋은 성적을 보여줄 수 있다.

<br>

### Model

<br>

GPT-2는 GPT-1에서 조금의 수정이 들어갔다고 한다.

Layer Normalization이 input의 각 sub-block으로 이동시키고, 추가적인 layer Normalization은 마지막 self-attention block에 추가하였다.

Residual Layers의 weights는 1/root(N)으로 scaling 했으며, 이때 N은 Residual Layer의 개수이다.

Vocabulary는 50,257개로 늘었으며 context size로 512에서 1024 tokens으로 늘고 batchsize도 512를 이용했다고 한다.

<br>

## Experiment

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a03cd050-a2e5-48b5-b976-094cee03ff4a" width='350' height='200'>

</p> 

<br>

<p align="center">


Table2에서 파라미터가 가장 작은 모델이 GPT-1이고 가장 큰 모델이 GPT-2이다.

345M 파라미터를 가진 모델은 BERT이다.

<br>

### Results

#### Language Modeling (Zero-Shot Setting)

<p align="center">

  <img src="https://github.com/user-attachments/assets/d8799616-5824-444c-afd3-41696dc49c02" width='700' height='300'>

</p> 

<br>

- PPL == Perplexity

해당 Table을 보면 Zero-shot setting인데도 불구하고 8개의 데이터셋 중 7개의 SOTA를 달성했고, 특히 dataset size가 작은 PTB, WikiText2에서 효과가 두드러지게 보인다.

<br>

#### Children's Book Test (Nouns Performance)

<p align="center">

  <img src="https://github.com/user-attachments/assets/fac77697-9e6a-4a0c-a51d-19e7255d8460" width='400' height='300'>

</p> 

<br>

Model Size가 커질수록 Performance가 증가하는 양상을 보이며 GPT-2는 SOTA를 달성했을 뿐만 아니라 인간의 능력에 필적하는 결과를 보인다.

<br>

#### LAMBADA (Long-Term Dependency)

<p align="center">

  <img src="https://github.com/user-attachments/assets/720c15ca-2e3b-4d95-8a64-0a1db00ce681" width='400' height='250'>

</p> 

<br>

Long-Term Dependency을 Benchmark 하기 위한 LAMBADA 데이터셋에서 GPT-2는 SOTA를 달성했다.

<br>

#### Winograd Schema Challenge (Inference Ability)

<p align="center">

  <img src="https://github.com/user-attachments/assets/a2947298-5c0c-428c-ad3a-7cf796d31427" width='400' height='350'>

</p> 

<br>

해당 실험은 Text의 Ambiguity를 푸는 작업을 진행해 Model의 inference ability를 평가한다.

GPT-2는 기존의 SOTA보다 7% 높은 ACC를 보여주었다.

<br>

#### Reading Comprehension (Reading Comprehension & QA)

CoQA라는 데이터셋을 사용하였으며 7개의 Domain의 Document에 대한 Question, Answer를 포함하고 있다.

이를 통해, Reading Comprehension 능력과 QA 능력을 평가할 수 있다.

SOTA 모델인 BERT에는 미치지 못했지만, GPT-2는 55라는 준수한 F1 score를 기록하였다.

GPT-2는 BERT와 달리 Labeled data를 사용한 것이 아니기에 더욱 고무적인 결과를 도출해냈다고 볼 수 있다.

<br>

#### Summarization

<p align="center">

  <img src="https://github.com/user-attachments/assets/bd3dd154-0fb8-4f97-a9b3-a88e7a71b799" width='400' height='300'>

</p> 

<br>

TL; DR은 문서 이후에 추가되는 토큰으로 Task-specific한 결과를 유도하기 위해 attach한 것이다.

해당 결과를 통해 얻을 수 있는 insight는 Task-specific한 결과를 유도한 토큰이 유의미한 역할을 한다는 것이다.

<br>

#### Translation

아쉽게도 GPT-2는 다른 Task에 비해 Performance가 조금 떨어진다는 결과를 확인할 수 있었다.

<br>

#### QA

<p align="center">

  <img src="https://github.com/user-attachments/assets/6e85fc24-1e4a-4f38-b11b-f2b5aaabb1c4" width='700' height='500'>

</p> 

<br>

QA task에서의 실험 결과이고 해당 결과에서 얻을 수 있는 insight는 QA task에서는 모델 사이즈가 매우 중요한 요인이라는 것이라고 한다.

<br>

#### Generalization vs Memorization

Train set과 Test set의 중복은 모델의 Memorization을 유도하고 Generalization 성능을 저하시킨다.

이러한 현상은 WebText 데이터셋에서도 나타날 수 있으며 WebText의 Overlap 정도는 아래와 같다.

<p align="center">

  <img src="https://github.com/user-attachments/assets/1666dc05-d126-47f1-acaa-f3df890af97c" width='700' height='200'>

</p> 

<br>

WebText의 train set와 test set의 성능 비교를 통해 Model Size에 따라 동시에 성능이 증가하는 것을 확인할 수 있다.

Memorization이 Model Performance 개선에 큰 요인은 아니었으며, 아직 모델이 underfitting되어 더 개선될 여지가 있음을 보여준다.

<p align="center">

  <img src="https://github.com/user-attachments/assets/25da8507-9d26-4166-9ac4-867723bfcf87" width='500' height='500'>

</p> 

<br>

## Conclusion

<br>

LLM을 충분히 크고 다양한 dataset으로 학습시킨다면 많은 domains과 datasets에서의 우수한 성능을 보여준다.

추가적으로 GPT-2는 GPT-1을 기반으로 해 Unsupervised pre-training을 극대화시켜 fine-tuning 없이 zero-shot setting에서 8개의 Dataset 중 7개의 Dataset의 SOTA를 달성해냈다.

<br>

