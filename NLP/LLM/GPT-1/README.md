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

TBU