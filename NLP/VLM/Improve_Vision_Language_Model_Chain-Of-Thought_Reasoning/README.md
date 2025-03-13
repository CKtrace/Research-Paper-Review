# [IMPROVE VISION LANGUAGE MODEL CHAIN-OF-THOUGHT REASONING](https://arxiv.org/pdf/2410.16198) (ICLR 2025)

<br>

## Abstract

<br>

 CoT는 VLM의 해석 가능성과 신뢰성을 개선시킬 수 있는 강력한 방법이다.
 
 그러나 현재 CoT는 최소한의 rationales를 지닌 짧은 annotations으로 구성되어 있는 dataset에 의존하기에 robust한 학습이 불가하다.

본 논문에서는 Short anwsers로 학습된 VLM이 reasoning task에서 generalize되지 않는 모습을 보여주며, 이는 더 많은 detailed responses가 요구된다.

이러한 문제를 해결하기 위해 제안하는 방법은 two-fold로 이루어져 있다.

1. 학습 데이터를 늘리고 VLM을 fine-tuning하기 위해 GPT-4o에서 rationale를 생성하였다.
2. Reasoning quality를 높이기 위해 정답과 오답 쌍으로 강화학습을 진행하였다.

본 연구에서는 학습에서의 detailed rationales의 필요성과 VLM의 reasoning capability 개선을 위한 강화학습을 강조한다.

<br>

## Introduction

<br>

VLM이 더욱 어려운 tasks에 적용되는 경우가 증가하면서 복잡한 문제로부터 근거를 도출하는 능력은 필수가 되었다.

그러나 현재 VLMs의 학습 접근 방식은 제한된 rationales와 짧은 answers가 지배적인 datasets에 의존하며, 이는 해당 task에 모델이 포괄적인 reasoning을 생성하는 능력에 제약이 된다.

본 연구에서는 이러한 limitations을 해결하는 방법인 Supervised finetuning(SFT), Reinforcement Learning(RL)을 소개한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/865f7c2a-197d-42e3-877b-086e98d924b9" width='700' height='400'>

</p> 

<br>

위의 figure에서 Question을 사람에게 물어보면 bar의 값들을 낱낱이 세서 합계를 낼 것이다.

그런데 이런 과정이 번거롭기 때문에 대부분 short answer인 '14'라는 답으로 annotated를 할 것이다.

결과적으로 data는 short answer와 minimal한 rationale으로만 이루어질 것이다.

본 논문에서 이와 같은 결과에 질문을 하나 던진다.

_direct prediction은 올바른 결과를 도출하기 위한 CoT를 수행할 수 있게 하는가?_

본 연구진들은 ChartQA 데이터셋에서 26k의 direct prediction으로 학습하였는데, direct predictions은 2.9%의 정확도가 증가하였지만, CoT prediction은 0.6%밖에 증가하지 못했다.

이러한 결과는 현재의 학습 접근 방식이 CoT reasoning의 개선 효과를 제한하고 있는 것을 보여준다고 한다.

그들은 CoT reasoning capabilites를 개선시키기 위해서는 detailed reasoning steps을 포함하는 data로의 학습이 요구될 것이라는 가설을 세웠다.

위에서 언급한 데이터를 만들기 위해 short ground truth annotations이 된 datasets을 활용하고 GPT-4o를 통해 correct answer로의 reasoning path를 생성하는 방법을 제안한다.

결과적으로 Supervised Fine-tuning(SFT)를 위한 193k의 CoT 예시들을 뽑아냈고 LLaVA-Reasoner-SFT 모델은 VLM CoT reasoning performance 개선을 보여주었다.

추가적으로 Rationales는 correct prediction ACC를 개선시키는 역할을 한다.

본 논문의 내용을 요약하면 아래와 같다.

(A) : 193k의 examples로 구성된 CoT dataset _SHAREGPT-4O-REASONING_ 는 다양한 VQA task에 적용 가능
(B) : 제안한 dataset으로 진행한 SFT의 효과적인 CoT reasoning 성능 개선을 입증함
(C) : DPO를 이용한 강화학습의 모델 성능 개선을 보여줌

<br>

## Method

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a9a0034b-734d-444a-90a5-35e0f48ee625" width='700' height='500'>

</p> 

<br>


