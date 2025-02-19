# [SELFCHECKGPT : Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/pdf/2303.08896)

<br>

## Abstract

<br>

LLMs은 사실에 대한 Hallucination이 존재하고 그들의 결과에 대한 신뢰를 훼손하는 거짓 정보들을 만들 수 있다.

현존하는 Fact-Checking 기법들은 ChatGPT와 같은 모델에서는 불가능한 output probability distribution 접근하는 방법 또는 외부 데이터베이스에 접근하는 방법이 있다.

본 논문에서는 Zero-Resource Fashion에서 간단한 샘플링 기법을 통해 Black-box 모델의 응답을 Fact-Checking하는 SELFCHECKGPT를 제안한다.


```
Zero-Resource Fashion -> i.e. without an external database
```

SELFCHECKGPT는 간단한 아이디어의 영향을 받았는데, 주어진 개념에 대한 샘플 응답들은 서로 유사하고 사실을 담고 있는 반면, Hallucinated fact는 샘플 응답들과 갈라지고 모순된다는 것이다.

해당 실험에서는 SELFCHECKGPT는 non-factual & factual sentences를 탐지하고, factuality라는 기준으로 응답에 순위를 매긴다.

<br>

## Introduction

<br>

LLM 모델들이 흔하게 허구의 정보를 만들어낸다. 

이러한 Hallucination을 막기 위해 존재하는 entropy나 token prob를 이용하는 unceratinty metrics는 API가 제공되지 않는 모델에 대해서는 불가능한 접근 방식이다.

이 문제에 대한 대안으로, fact-verification 기반 접근이 있는데 해당 접근은 외부 database로부터 generation을 평가하는 score를 찾는 방법이다.

허나, 해당 방법도 database에 해당 정보의 유무로 평가되기에 절대적 평가 지표가 될 수 없다.

본 논문에서 제안하는 SelfCheckGPT는 sampling 기반의 접근으로 LLM이 hallucinated인지 factual한지 판단한다. 

SelfCheckGPT는 첫 번째 LLM 응답의 hallucination을 분석하는 모델이자 첫 번째 black-box 모델에서의 zero-resource hallucination detection의 해결책이다.

해당 모델은 샘플링한 여러 응답을 바탕으로 다른 응답들과 비교를 통해 정보 일관성을 확인하고 factual인지 hallucinated인지 확인한다.

SelfCheckGPT는 샘플 응답들에 대해서만 영향을 받기에, black-box 모델들에 적용할 수 있다는 장점뿐만 아니라 external database가 요구되지 않는다는 장점 또한 존재한다.

SelfCehckGPT는 다섯 가지 사항을 고려한다 : BERTScore, QA, n-gram, NLI(자연어 추론), LLM prompting.

해당 모델은 LLM의 중요한 문제의 강력한 첫 번째 baseline 될 뿐더러, grey-box method보다 우수한 hallucination detection method이다.

<br>

## Background and Related Work

<br>

```
- Hallucination of Large Language Model

- Sequence Level Uncertainty Estimation

- Fact Verification -> Using External Database

- Grey-Box Factuality Assessment 
```

<br>

## Grey-Box Factuality Assessment

<br>

### Uncertainty-based Assessment

<br>

Flat probability distribution을 갖는 결과는 hallucination을 가져온다.

해당 Insight는 uncertainty metrics와 factuality의 connection을 이해하게 한다.

Factual 문장은 higher likelihood와 lower entropy를 지닌 토큰들을 포함하고 있는 경향이 있다.

반면에 hallucination은 high uncertainty를 갖는 flat probability의 position들로부터 나온다.

<br>

### Token-level Probability

<br>

```
R : LLM's Response
i : R in i th sentence
j : i in j th token
J : Number of tokens in sentence

Avg(-log p) = - (1/J)SUM(logp_ij)
MAX(-log p) = max(j) (-logp_ij)
```

<br>

### Entropy

<br>

```
The entropy of the output distribution is:

H_ij = - SUM(w^~ ∈ W) p_ij(w^~)logp_ij(w^~)
```

p_ij(w^~)는 w^~라는 단어가 i번 째 문장에 j번 째 토큰이 생성될 확률이며, W는 Vocabulary에 담겨 있는 모든 가능 단어들의 집합이다.

<br>

```
Two Entropy-based metrics

Avg(H) = (1/J)SUM(H_ij)
MAX(H) = MAX(j) (H_ij)
```

<br>

## Black-Box Factuality Assessment

<br>

Grey-Box method는 API 호출이 가능한 LLM에만 가능해 token-level information과 같은 정보들을 얻지 못한다는 단점이 존재한다.

그래서 본 논문에서는 Black-Box에서의 접근 방식을 고려한다.

<br>

### Proxy LLMs

<br>

Grey-box 접근과 유사한 간단한 접근은 porxy LLM을 사용하는 것이다.

```
proxy LLM은 black-box LLM이 생성하는 text의 token-level 확률에 근사하는 값을 뽑아내는데 쓸 수 있다.
```

<br>

## SelfCheckGPT

<br>

본 논문에서 제안하는 SelfCheckGPT는 LLM의 여러 샘플 응답들을 비교하고 일관성을 측정하는 방식으로 black-box zero-resource hallucination detection을 수행한다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/c9109e90-1711-44e9-b188-8451e58fa40a" width='500' height='550'>

</p> 

<br>


```
Notation

R : LLM reponse

N : Stochastic LLM response samples {S^1, S^2, ..., S^N} (Using Same Query)

R <-> N : Measuring Consistency

S(i) :  Hallucination score of i-th sentence -> S(i) ∈ [0.0, 1.0]
        S(i) -> 0 : Factual     S(i) -> 1 : Hallucinated 
```

<br>

### SelfCheckGPT with BERTScore

<br>

```
B : RoBERTa-Large로 계산한 BERTScore between two sentences
r_i : R에 있는 i-th Sentence
s_k^n : n번 째 Sample Response의 k번 째 Sentence
```

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/87958d05-afc0-49a7-a9a0-3acb147ade72" width='300' height='100'>

</p> 

<br>

해당 방법은 문장에서의 정보가 여러 샘플들에서 많이 나타면 해당 정보는 factual하다고 판단하고, 어떤 정보가 다른 샘플에서 등장하지 않다면 hallucination이라고 판단한다.

<br>

### SelfCheckGPT with Question Answering

<br>

SelfCheckGPT에서 일관성을 측정하기 위해 MQAG(Multiple-choice Question Answering Generation) Framework를 사용하였다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/dad6334b-fa9a-4915-902b-2552e65ab621" width='700' height='500'>

</p> 

<br>

```
QG -> G, QA -> A

R : Response
r_i : i-th sentence in Response
q : questions
o : options
a : answer
```

<br>

#### Stage G

<br>

G는 G1, G2 두 단계로 나눌 수 있다.

G1은 q,a pair를 이용한 generate, G2는 o\a(오답)을 generate.

최종 선택지는 o = {a, o\a} = [o1, o2, o3, o4]가 된다.

또한 G에서 unanswerable(Bad) question을 filter  out하기 위해 answerability score alpha를 설정한다.

alpha = 0.0이면 q는 unanswerable, 1.0이면 answerable한 것이다.

<br>

#### Stage A

<br>

```
a_R : Main Response가 주어졌을 때 q의 정답 o

a_S^n : n-th sampled response S^n이 주어졌을 때 q의 정답 o

N_m : a_R == a_S^n의 개수

N_n : a_R != a_S^n의 개수
```

위처럼 N개의 sampled response에 대해 반복한 뒤, N_m과 N_n을 이용해 S_QA(i, q)가 i-th sentence에 대한 hallucination score이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/59327907-72a6-4a1d-bd28-3e965e153741" width='100' height='50'>
Original S_QA 값
</p> 

<br>

본 논문에서는 soft-counting을 적용한 inconsistency score를 아래와 같이 수정하였다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/f2d7bde8-b3d3-40bf-9c10-a19459e8c256" width='250' height='100'>
Modification S_QA 값
</p> 

<br>

```
L_m : N_m

L_n : N_n

B_1 : P(a` != a_R | F)

B_2 : P(a = a_r | T)

γ1 : B_2 / (1 - B_1)

γ2 : B_1 / (1 - B_2)

F: Non-Factual

T : Factual
```

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/5ccc03c2-82c0-4dac-a9fb-cbd14ad2d311" width='500' height='200'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/7751dd9c-5319-420b-9822-9121f60033a1" width='500' height='200'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/707a03a8-f856-4e76-b061-bc4ad1909d7f" width='500' height='200'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/6b280e57-5a2d-44ed-a24f-3b51b5dc1011" width='500' height='200'>

</p> 

<br>

이제 마지막으로 answerability를 반영해주는 경험적으로 alpha 이하의 점수를 가지는 question은 기존 방법처럼 filter out하기보다 alpha를 이용해 N_m과 N_n을 각각 N_m', 
N_n'으로 soft counting 하는 것이 성능 향상에 도움이 된다고 한다.

a_n은 정의에 따라 L_m, L_n에 있는 ground-truth이고, alpha_n은 question에 대한 answerability score이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/c3b4296b-814e-4924-8ac6-e763f0366ef4" width='500' height='100'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/10688af9-f0a4-4675-88ce-08fbaae67c4c" width='500' height='100'>

</p> 

<br>

## SelfCheckGPT with n-gram

<br>

LLM's token probability에 근사 가능한 n-grem Language model을 만든다.

