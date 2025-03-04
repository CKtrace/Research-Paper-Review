# [A Survey of Hallucination in "Large" Foundation Models](https://arxiv.org/pdf/2309.05922)(arXiv 2023)

<br>

## Abstract

<br>

Foundation Model(FM)에서의 Hallucination은 사실과 벗어나거나 만들어진 정보가 담긴 content를 generation 하는 것이다.

본 논문에서는 Large Foundation Model(LFM)에서 볼 수 있는 다양한 유형의 hallucination 현상을 분류하고 hallucination 정도를 나타내는 평가 기준을 제시한다.

또한 LFM에서 존재하는 hallucination 개선 전략과 해당 분야에서의 잠재력 있는 방향성을 제시한다.

<br>

## Introduction

<br>

"Foundation Model"이란 광범위하고 다양한 라벨링 되지 않은 데이터로 학습해 여러 general task를 수행 가능한 모델을 말한다.

Hallucination은 model이 fictional, misleading, entirely fabricated한 details, facts, claims을 포함하는 text를 생성할 때 발생할 수 있다.

최근 LFM은 산업 분야, 학업 분야 등 여러 분야에서 쓰이면서 hallucination에 대한 관심은 부상 중이다.

LLM에 대한 hallucination에 다룬 연구들이 많았지만, 최근 VLM에서의 hallucination을 다루는 연구는 드물며 본 연구에서는 VLM에서의 hallucination에 대해 다루고자 한다.

<br>

## Hallucination in Large Language Models

<br>

Hallucination은 LLM이 response를 만들어낼때 발생한다.

<br>

### Hallucination datasets

<br>

[HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2305.11747) 논문에서 만든 HaluEval이라는 데이터셋이 존재한다.

LLM에서는 아래와 같은 hallucination 접근 방식들이 존재한다.

1. Hallucination mitigation using external knowledge
2. Hallucination mitigation using prompting techniques

<br>

### Domain-specific LLMs

<br>

#### Medicine

<br>

[Med-HALT: Medical Domain Hallucination Test for Large Language Models](https://arxiv.org/pdf/2307.15343)

<br>

#### Law

<br>

[Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model](https://arxiv.org/pdf/2306.16092)


<br>

## Hallucination in Large Models

<br>

[Hallucination Improves the Performance of Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2307.12168)에서 제안한 Hallucinator를 이용하기 위해서는 두 가지 조건을 충족해야 한다.

1. 충분한 positive pairs가 보존되어야 한다.
2. 충분한 variation이 존재해야 한다.

이 두 가지 조건을 충족하지 못하면, 해당 framework는 오버피팅 되기 쉬울 뿐더러, 의미 있는 semantic을 구분하는 능력이 저하될 것이라 한다.

SOTA를 달성한 LVLM(Large Vision Language Models)은 높은 비율의 hallucinatory text가 존재하는데 이는 30% 정도이며 존재하지 않는 objects, 부정확한 descriptions, erroneous relationships가 포함되어 있다.

이러한 문제점을 바로 잡고자, MHalDetect 데이터셋이 소개되었고 이는 모델이 hallucination을 detecting and preventing하기 위해 학습하고 평가할 수 있게 설계되었다.

해당 데이터셋은 VQA example로 구성되어있다.

이후 내용은 Video 부분, Audio 부분에 대해 설명한다.

<br>

## Hallucination is _not_ always harmful : A different perspective

<br>

Hallucination을 창의적으로 활용하면, 하나의 데이터에서 나올 수 없는 combination of idea를 생성해낼 수 있다.

특정 전문적인 분야의 지식을 제공하는 LLM에서는 Hallucination은 critical 하지만, creative or artistic한 노력이 요구되는 곳에서는 Hallucination이 이끌어내는 한 번도 보지 못했던 generation output이 장점이 될 수 있다.

<br>

## Conclusion and Future Directions

<br>

### Automated Evaluation of Hallucination

<br>

1. Development of Evaluation Metrics

2. Human-AI Collaboration

3. Fine-Tuning Strategies

<br>

### Imporving Detection and Mitigation Strategies with Curated Sources of Knowledge

<br>

1. Knowledge Graph Integration

2. Fact-checking and Verification Models

3. Bias Detection and Mitigation

4. Active Learning

5. Ethical Guidelines and Regulation