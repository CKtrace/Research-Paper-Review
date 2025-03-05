# [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2305.10355) (EMNLP 2023)

<br>

## Abstract

<br>

최근 LVLM은 성능이 뛰어난 LLMs을 통합하여 복잡한 multi-modal tasks에서의 성능 개선을 해왔다.

이러한 긍정적 진행에도 불구하고, 본 연구에서는 LVLM이 object hallucinations을 겪는다는 것을 확인했다.

이것을 조사하기 위해, 본 논문은 처음으로 LVLMs의 object hallucination을 체계적인 연구를 제안한다.

본 논문에서는 여러 평가 실험을 진행하고, LVLMs이 흔히 겪는 object hallucination issues를 소개한다.

또한, 본 연구진들은 visual instructions이 hallucination에 영향을 준다는 내용에 대해 의논했으며, 찾은 내용은 다음과 같다.

```

Visual Instruction이 모델이 해석할 때 영향을 미쳐, 특정 objects가 자주 등장하거나 image에 있는 objects와 동시 발생하는 경우 명백하게 LVLMs에 의해 halllucinate 된다는 사실이 증명됐다.

(이때, Visual Instruction은 Prompt에서 이미지에 대한 정보를 기입한 것을 말하는 것으로 보임.)

```

게다가 본 연구에서는 POPE라는 polling-based query method를 사용해 object hallucination evaluation을 성능을 개선 시켰다.

<br>

## Introduction

<br>

LLM의 성능이 나날히 발전하면서, Visual semantic Understanding을 향상시키기 위해 우수한 LLM을 기반으로 한 Multi-Modal Model들이 연구되기 시작했다.

이러한 연구들의 흐름은 우수한 성능을 보이는 LLM을 포함함으로써 vision-language pre-trained model(VLPM)의 성능 개선시켜 나갔으며 이러한 모델 구조를 Large vision-language model(LVLM)이라고 부른다.

LVLM의 끊임없는 발전에도 불구하고, LLMs과 VLPMs는 Hallucination을 겪는다는 것은 알려져 왔다.

특히, LLM은 unintended text에 hallucinate되고 VLPM은 image에서 존재하지 않는 object를 만들어내는 양상을 보인다.

본 논문에서는 LVLM의 object hallucination 정도를 체계적으로 평가한다.

제안한 연구를 수행하기 위해 CHAIR(Caption Hallucination Assessment with Image Relevance) metric을 사용해 MSCOCO dataset에서의 LVLMs의 hallucination 정도를 파악한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/93baadde-fb07-40e2-830c-262be4be9cc9" width='350' height='500'>

</p> 

<br>

해당 실험을 통해 LVLM은 small vision-language models보다 hallucination 정도가 큰 것을 확인할 수 있었다.

또한 현재 존재하는 object hallucination evaluation method(CHAIR)은 최적합이 아니라고 판단하여 POPE Method를 제안한다.

POPE Method의 간단한 아이디어는 evaluation of hallucination을 이진 분류 task로 전환하는 것이다.

```
LVLMs 프롬프트 설계 시, yes or no로 답할 수 있게 짧은 질문 형식을 채택하는 것이다.
e.g.,  Is there a car in the image?
```

해당 방법은 더욱 stable and flexible하다고 한다.

POPE는 unannotated datasets까지 쉽게 확장할 수 있다는 장점 또한 존재한다.

<br>

## Background

<br>

### LVLM

<br>

Training of LVLMs은 세 가지 Steps으로 나뉜다.

1. Vision Encoder와 Language Encoder를 Large-Scale unimodal data(image & text data)로 pre-training한다.
2. 두 encoders가 image-text alignment pre-training을 통해 정렬되는데, 이때 LLM은 주어진 image의 의미있는 caption을 생성한다.
3. 정렬된 model은 image-text instructions에 맞게 fine-tuned되며, 정답을 생성해낸다.


Visual Encoder와 LLM이 잘 정렬된다면 LVLM은 우수한 visual understanding ability를 이뤄낼 수 있다.

이는 Visual Semantic만 잡을 수 있는 것이 아니라, linguistic semantics까지 깊게 이해할 수 있게 된다.

게다가 LVLM은 Objects에 대한 개념 관련 complex resoning을 수행함으로써 다양한 multimodal tasks에서 성능 개선을 이뤄낸다.

<br>

## Object Hallucination

<br>

비록 LVLM이 vision-language tasks에서 좋은 성적을 보이지만, VLPMs과 같이 object hallucination issue를 겪는다.

Object Hallucination이란 model이 target image에 대해 올바르지 않은 descriptions이나 captions을 생성하는 것을 말한다.

LVLMs이 real-world에 적용될 때, 예를 들어, 자율주행 차량에서 object hallucination이 존재한다면 예상치 못한 events를 만날 수 있게되며 이는 심각한 안전 문제와 직결된다.

이러한 issue를 완화하기 위해 본 연구에서는 LVLMs의 object hallucination 존재 evaluation 방법을 제안한다.

<br>

### Evaluation Settings

<br>

Caption Hallucination Assessment with Image Relevance (CHAIR) image captioning tasks 유명한 tasks이다.

CHAIR는 image가 아닌 caption에서의 hallucination된 비율을 나타낸다.

본 연구에서는 두 가지 변수의 CHAIR를 사용했다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/17364db3-c24b-4f2e-b0b1-71cab922d5b5" width='400' height='150'>

</p> 

<br>

본 실험에서는 다음과 같은 5가지 LVLMs을 채택했다 : mPLUG-Owl, LLaVA, Multimodal-GPT, MiniGPT-4

```
Table 1을 보면 I_1, I_2가 있다.

I_1 : Generate a short caption of the image
I_2 : Provide a brief description of the given image
```

그리고 각각 captions에 대한 CHAIR를 구해 Table1과 같은 결과를 얻어냈다.

<br>

### Evaluation Results

<br>

#### Severity of hallucinations

<br>

Table 1을 보면 InstructBLIP이 가장 좋은 CHAIR 성능을 보인다.

가능성이 높은 이유는, 연관이 서로 적은 넓고 다양한 datasets로부터 수집된 visual instructions을 채택하는 InstructBLIP과 달리, 다른 LVLMs은 대부분 unimodal LLMs이 생성해낸 visual instructions을 채택한다.

LLMs에서 생성해낸 visual instructions은 더욱 길고 정보가 많아 일반적으로 좋지만, LLM의 hallucination으로 unexpected descriptive information이 담길 수도 있다.

<br>

#### Disadvantages of CHAIR

<br>

CHAIR metric을 사용하면 모델이 서로 다른 instruction을 택했을 때(I_1, I_2) CHAIR metric은 불안정하다.

따라서 LVLM의 object hallucination을 안정적이고 편하게 구할 수 있는 방법이 요구된다.

<br>

## Influence of Instruction Data on Object Hallucination

<br>

본 section에서는 visual instruction data의 영향을 조사해본다.

<br>

### Hypotheses

본 연구에서 세운 가설은 두 가지가 존재한다.

1. Visual Instruct Datasets에서 자주 나오는 ojects이면 hallucinate될 확률이 높을거다.
2. Co-occuring objects Frequency가 높을수록 halllucinate될 확률이 높을거다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/eab85e53-b134-4995-bd49-2e563bc265db" width='700' height='500'>

</p> 

<br>

Figure 2의 (a)는 MSCOCO에서 Frequency가 높은 10개의 objects의 hallucinatation 횟수를 나타낸 것이고,
Figure 2의 (b)는 MSCOCO에서 'dining table'과 Co-occuring Frequency가 높은 10개의 단어들의 halllucination 횟수를 나타낸 것이다.

해당 결과를 통해 두 가지 가설을 증명할 수 있었다.

<br>

### Quantitative Analysis

<br>

위에서 얻은 지식들을 통합하기 위해, 등장 빈도와 hallucination 횟수의 consistency를 측정하기 위해 top-k hit ratio(HR@k)를 채택했다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/e6c24131-a2fe-448e-bdc5-59c76a9c2e75" width='400' height='100'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/0eba8ef4-972e-4411-9631-c2a8868e6d96" width='400' height='100'>

</p> 

<br>

Hallucinated(i)는 i-th example의 hallucinated objects의 개수를 뜻하고, Hit@k(i)는 top-k frequency 횟수를 나타낸다.

또한, Hit@k(i, o)는 object o와 동시발생하는 top-k frequency 횟수를 나타낸다.

<br>

## POPE

<br>

Polling-based Object Probing Evaluation (POPE)는 LVLMs의 Hallucination을 간단하고 효과적으로 평가할 수 있는 접근 방법이다.

POPE Pipeline은 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/365447a0-0aa7-455e-b4f6-feaf8e11a8aa" width='700' height='400'>

</p> 

<br>

이때 Automatic Segmentation tools은 SEEM을 사용하였다고 한다.

또한 해당 파이프라인에서는 아래와 같은 세 가지 세팅으로 실험을 진행했다.

- Random Sampling : Image에 없는 Object random sample
- Popular Sampling : 전체 데이터셋의 object 중 Top-k most freqeunt object 중 현재 Image에 없는 object
- Adversarial Sampling : Rank all objects according to their Co-occuring frequencies with the ground-truth objects -> Select top-k freqeunt ones(현재 Image에 없는 Object)

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/0e59bf9a-ca70-447b-af74-ea398d724c14" width='700' height='400'>

</p> 

<br>

### Advantages of POPE

<br>

기존의 CHAIR로 LVLM을 평가할 시, LVLM은 프롬프트에 민감하고 object annotation이 요구되며 수동으로 평가를 위한 규칙을 지정해줘야 했다.

하지만 POPE를 사용할 경우 _Stability_, _Scalability_(Without annotation), _consistency_(Yes/No 응답의 일관성)

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/8dbde542-b4c7-4e72-83dc-378688f23750" width='700' height='200'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/0afa9785-7a32-4eaf-9de3-d8764ac6d7d2" width='700' height='300'>

</p> 

<br>

## Conclusion

<br>

본 논문에서는 input instructions과 LVLM에서 생성된 text로 인해 hallucinatation evaluation method가 영향을 받아 evaluation result의 신뢰도가 떨어지는 문제를
POPE를 통해 해결하고자 했다.

<br>