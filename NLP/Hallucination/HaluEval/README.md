# [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2305.11747) (EMNLP 2023)

<br>

## Abstract

<br>

Chat-GPT와 같은 LLM은 hallucinations을 생성해내는 경향이 존재한다.

어떤 유형의 content, LLM이 어느 정도의 hallucination 하는 경향이 있는지 확인하기 위한 HaluEval dataset을 제안한다.

본 데이터셋은 사람이 annotate한 hallucinated samples로 구성되어 있으며 이는 LLM의 hallucination 인식 성능을 평가할 수 있다.

Hallucinated smaples를 자동으로 생성하기 위한 framework를 제안하는데, 이는 _sampling-then-filtering_ 인 두 가지 step으로 구성되어 있다.

추가적으로, 사람을 고용해서 Chat-GPT에서 생성해내는 hallucination을 annotated하게 하였으며, 특정 topic에서 응답의 약 19.5%의 hallucinated response를 하는 것을 확인했다.

또한 현존하는 LLM은 text에서 hallucination을 인식하는 것에 어려움을 겪고 있다.

그러나 본 연구에서는 외부 지식 또는 reasoning step을 추가하는 것이 LLM이 hallucination을 인식하는데 도움이 된다는 것을 증명한다.

<br>

## Introduction

<br>

LLM의 눈부신 발전이 이어지는 와중에, LLMs은 hallucination을 겪는 모습을 보여주었으며 이는 real-world에 적용되었을 때 잠재적인 위험을 지니고 있다.

대부분의 Hallucination 연구는 specific tasks와 small language models에서의 hallucination이 발생하는 원인에 대한 내용이다.

하지만, 여전히 어떤 유형의 content 그리고 LLM이 어느 정도의 hallucination 하는 경향을 보이는지에 대해서는 해결되지 않았다.

이러한 문제를 해결하고자 HaluEval dataset을 제안하며, 해당 데이터셋은 LLM을 분석하고 평가하기 위한 35,000개의 hallucinated/normal samples로 이루어져있다.

HaluEval은 5,000개의 user queries와 Chat-gpt의 응답 세트와 30,000개의 세 개의 task(QA, knowledge-grounded dialogue, text-summarization)의 task-specific examples로 구성된다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/3f1efcdd-159e-4585-972a-e5a842cfeac9" width='700' height='350'>

</p> 

<br>

위 Figure는 Construction pipeline of HaluEval이며 어떤 식으로 HaluEval이 생성되었는지 알 수 있다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/42383908-31b7-40d6-bc6c-9bbcaf686d98" width='400' height='300'>

</p> 

<br>

위 테이블의 User query는 human-annoted query로써, hallucinated response를 분석해 어떤 content에서 LLM이 hallucinate되는 경향이 있는지, 더 나아가 hallucinate 정도를 덜어내는 방법을 고안하게 할 수도 있다.

게다가, task-specific examples에서는 hallucinated samples을 생성하기 위한 automatic two-stage approach를 했다.

1. 존재하는 task dataset를 seed data로 삼고 두 가지 스타일의 instruction으로 ChatGPT를 다양한 측면에서의 hallucinated samples를 생성하도록 했다. (One-pass Instruction & Conversational)

2. LLM이 평가하기 어렵고 그럴듯한 hallucinated sample을 선택하기 위해 실제 답으로 정교하게 만든 filtering instruction을 이용했고 ChatGPT가 sample을 선택하게 하였다.


또한 _sampling-then-filtering_ approach를 제안하는데, 이는 각 specific task에 대해 hallucinated counterpart를 생성하게 한다.

이를 통해 만들어진 hallucinated samples은 LLM의 hallucination Recognition과 Analyze하는 능력을 시험해 볼 수 있다.

HaluEval에서의 LLM의 성능을 보다 더 잘 이해할 수 있게 본 연구진들은 성능 좋은 여러 LLM으로 실험들을 수행했다고 한다.

그들의 Key findings을 요약하면 아래와 같다.

```
1. ChatGPT는 검증 불가한 정보를 제작하여 response로 제작하는 경향을 보이며 이는 약 응답의 19.5%를 차지한다.

2. 현존하는 LLM은 생성된 text에서의 hallucination을 인식하는데 어려움을 겪고 있다.

3. 외부 지식이나 resoning steps을 통해 LLMs의 부족한 recognizing hallucinations 성능을 개선시킬 수 있다.
```


<br>

## The HaluEval Benchmark

<br>

HaluEval의 목표는 어떤 유형의 contents 그리고 어느 정도의 hallucination 정도를 LLM이 겪는지를 확인하는 것이다.


### Automatic Generation

<br>

제안하는 생성 pipeline은 two-step으로 구성된다.

1. 다양한 hallucination sampling
2. High-Quality hallucination filtering

위와 같은 pipeline을 자동 생성하기 위해 ChatGPT를 이용했다고 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/6bdbf73f-68d3-4245-bdd8-dccc0a841a36" width='700' height='550'>

</p> 

<br>

#### Diverse Hallucination Sampling

<br>

앞서 말한 ChatGPT를 이용한 두 가지 전략이 있다.

첫 번째는 _one-pass instruction_ 이며 ChatGPT에 complete instruction을 주어 hallucinated answer를 생성한다.

두 번째는 _conversationale_ 이며 ChatGPT를 instruction을 배울 수 있게 하고 그것을 마스터할 수 있게 지도한다.

위 두 가지 전략을 통해 각 질문에 대해 다양한 hallucinated answers를 생성할 수 있으며, 이는 추후에 plausible하고 difficult한 것 하나가 선택될 것이다.

<br>

#### Instruction Design

<br>

본 논문에서 제안하는 접근법의 핵심은 ChatGPT를 hallucinated samples를 생성하기 위한 효과적인 instruction을 design하는 것이다.

Hallucination sampling instruction은 세 가지 중요한 파트로 구성되며 Table 2에서 이를 확인할 수 있다.

1. Intention Description
2. Hallucination Pattern
3. Hallucination Demonstration

Intention Description에서는 system의 역할을 지정해주고 input과 generation 목적을 정의해준다.

Hallucinated samples의 type과 quality를 조절하기 위해 hallucination pattern과 demonstration을 소개한다.

Few-shot demonstrations은 system이 hallucination pattern을 이해하는 것을 돕는다.

QA, Knowledge-grounded dialogue, text summarization : 이 세 가지 task에 대한 hallucinated samples를 생성한다.

QA에서는 네 가지 유형의 hallucination patterns으로 구성했다 : Comprehension, Factualness, Sepcificity, Inference

Knowledge-grounded dialogue에서는 세 가지 유형의 hallucination patterns으로 구성했다 : extrinsic-soft, extrinsic-hard, extrinsic-grouped

Text Summarization에서는 세 가지 유형의 hallucination patterns으로 구성했다 : Factual, Non-Factual, Intrinsic

위 세 가지 tasks를 위해 HotpotQA, OpenDialKG, CNN/Daily Mail의 training set에서 30,000개의 instances를 랜덤으로 선택하고 해당 data들에 대한 hallucinated examples를 생성했다.

<br>

### High Quality Hallucination Filtering

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/43e34472-f774-420c-a80e-721781b7423c" width='700' height='550'>

</p> 

<br>

Most plausible and difficult한 hallucination samples을 선택하기 위해 두 가지 방법의 sampling을 사용한다.

Table3에서 보다시피, ground-truth answers로 개선시킨 instruction of hallucination filtering을 design해 두 개의 hallucinated candidates 중 더 나은 하나를 선택할 수 있게 한다.

위와 같은 _sampling-then-filtering_ process를 통해 30,000개의 세 개의 tasks에서 hallucinated samples를 만들어 낼 수 있었다.

<br>

### Human Annotation

<br>

Hallucinated samples 생성하기 위해 추가적으로 human labelers도 고용하여 ChatGPT의 responses가 hallucinated content를 포함하는지 annotate하게 했다. 

<br>

### Conclusion

<br>

타 모델들과 달리 GPT-4가 가장 정확한 답변들을 내놓았지만, 여전히 Hallucination problem은 해결되지 않았다.

1. Hallucination 문제 해결을 위한 연구의 필요성 강조
2. HaluEval Dataset을 통한 향후 LLM Hallucination 연구에 기여 

Hallucination은 세 가지 측면으로 구성된다.

1. Unverifiable
2. Non-factual
3. Irrelevant

<br>

### Labeler Details

<br>

