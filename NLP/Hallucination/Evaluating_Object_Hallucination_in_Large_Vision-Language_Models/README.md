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

### Object Hallucination

<br>

비록 LVLM이 vision-language tasks에서 좋은 성적을 보이지만, VLPMs과 같이 object hallucination issue를 겪는다.

Object Hallucination이란 model이 target image에 대해 올바르지 않은 descriptions이나 captions을 생성하는 것을 말한다.

LVLMs이 real-world에 적용될 때, 예를 들어, 자율주행 차량에서 object hallucination이 존재한다면 예상치 못한 events를 만날 수 있게되며 이는 심각한 안전 문제와 직결된다.

이러한 issue를 완화하기 위해 본 연구에서는 LVLMs의 object hallucination 존재 evaluation 방법을 제안한다.
