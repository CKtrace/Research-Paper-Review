# [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971) (arXiv 2023)

<br>

## Abstract

<br>

본 논문에서는 LLaMA 모델을 제안하며, 7B부터 65B 모델까지 있다.

본 모델은 1조 개의 tokens으로 학습을 진행했으며, 이는 공적으로 사용가능한 데이터셋으로만 SOTA 모델로 만들어냈으며 소유권이 있는 데이터셋이나 접근 불가한 데이터는 사용하지 않았다.

LLaMA-13B는 GPT-3를 대부분의 Benchmarks에서 뛰어넘는 성적을 보여주었으며, LLaMA-65B는 Chinchilla-70B와 PaLM-540B와 견줄 수 있다.

<br>

## Introduction

<br>

LLM의 Parameters가 클수록 더욱 나은 performance를 보인다는 것이 여러 연구를 통해 알려졌다.

하지만, 최근 연구에서 작은 모델을 더욱 많은 데이터로 학습시킬 시 best performance를 얻어낼 수 있다는 것이 밝혀졌다.

해당 연구에서는 특정 _training_ compute budget에 대해 dataset과 model size를 scaling 하는 방법을 정의해냈지만, LM scale 시 치명적인 _inference_ budget에 대해서는 신경쓰지 못했다.

기존의 연구에서는 10B 모델은 200B 개의 tokens으로 학습시키는 것을 추천했지만, 본 연구에서는 1T 개의 tokens으로 학습 시켜도 성능이 개선되는 것을 찾았다고 한다.

본 연구에서 제안하는 LLaMA는 7B ~ 65B 파라미터를 갖는 모델이며 LLaMA-13B는 GPT-3보다 대부분의 Benchmark에서 좋은 성능을 보였으며, 모델 사이즈는 10배 더 작다.

추가적으로 65B LLaMA는 Chinchilla 모델이나 PaLM-540B와 비교했을 때 경쟁력이 있었다고 한다.

Chinchilla, PaLM, GPT-3와 달리 LLaMA는 오로지 공공의 데이터만 학습하는데 사용했고, 접근 불가하거나 소유권이 존재하는 데이터는 사용하지 않았다고 한다.

LLaMA는 Transformer 요소를 수정해서 사용했다고 한다.

<br>

## Approach

<br>

LLaMA는 GPT-3와 Chinchilla의 scaling laws에 영향을 받았다고 한다.

<br>

### Pre-training Data

<br>

본 파트에서는 학습에 사용된 데이터들에 대해 설명하고 있다.
(각 데이터들에 대한 상세 정보는 논문을 참고하자.)

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/2e78ffd5-0c1e-4ae8-9cc1-717cb54317e3" width='400' height='400'>

</p> 

관심깊게 볼 점은, Wikipedia와 Books data를 제외한 다른 데이터들은 Epoch을 1 정도만 학습을 진행한 점이다. 

Wikipedia와 Books data도 2 정도의 Epoch만으로 학습을 진행했다.

추가적으로 Tokenizer는 SenetencePiece에서 구현된 BPE algorithm을 택했다.

<br>

### Architecture

<br>

본 파트에서는 기존 구조와 차별점을 갖는 것들에 대해 설명한다.

<br>

#### Pre-Normalization [GPT-3]

Training Stability 개선을 위해 각 transformer sub-layer의 output Normalize을 하지 않고, input Normalize을 진행했다.

이때, Normalizing function은 RMSNorm을 사용했다.

<br>

#### SwiGLU activation function [PaLM]

성능 개선을 위해 Activation Function을 ReLU 대신 SwiGLU를 사용했다.

PaLM에서 사용한 방법과 동일하게 GLU의 dimenstion을 4d가 아니라 2/3 * 4d를 사용하는 것으로 수정했다고 한다.

<br>

#### Rotary Embeddings [GPTNeo]

Absolute positional embeddings를 제거하고 rotary positional embeddings(RoPE)로 대체했다.

<br>

### Optimizer

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/9776b22f-8923-4442-b069-725ff89292a9" width='700' height='200'>

</p> 

<br>

```
- AdamW
- Cosine LR Schedule : final LR = 10% of MAX
- Weight Decay = 0.1
- Gradient Clipping = 1.0
- Warmup steps : 2,000
```

<br>

#### Efficient implementation

<br>

제안한 모델의 학습 속도를 개선하기 위해 몇 가지 optimizations을 만들었다고 한다.

첫 번째는, Casual multi-head attention을 사용해 memory 사용량과 runtime 감소를 시켰다.

backward에서는 Flashattention을 사용했으며, 이는 attention weights를 저장하지 않고 LM task에서 masked 되는 부분은 key와 query를 계산하지 않게 한다.

두 번째는, backward pass(forward pass 이후 역으로 미분해가며 기울기 값들을 구해가는 과정)에서 다시 계산되는 activations의 양을 감소시켰다.

이는, Linear layers의 output과 같은 계산량이 많은 activations을 저장해놓는 방식이며, pytorch autograd 대신 직접 backward function을 구현했다고 한다.

이 뿐만 아니라, model & sequence 병렬화를 적용해 Memory usage 감소, activation 계산과 GPU 병렬 처리에서의 각 GPU의 communication을 최대한 overlap 시켜 최적화하는 방법도 사용하였다.

이러한 Efficient implementation의 결과로 LLaMA-65B를 1.4T tokens을 학습시키는데 21일이 소요된다고 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/1dccba33-7da0-46cf-b69e-a5623d23cb52" width='400' height='500'>

</p> 

<br>

## Main results

<br>

해당 섹션에서는 여러 Benchmark에서의 LLaMA 성능을 보여주고 있다.

```
각각의 Benchmark에 대해서는 논문을 참고해주길 바라며, 알고 가야할 부분에 대해 작성하도록 한다.
```

본 논문에서는 LLaMA를 전체 중 20개의 Benchmarks에 대한 결과를 보여주고 있다.

### Zero-shot
task와 test example에 대한 textual description을 제공한다.

모델에 또한 open-ended generation(개방형 텍스트 생성 : 주어진 context에 이어질 문장 생성) 또는 제공된 답변에 대해 ranks를 매기는 방식을 제공한다.

<br>

### Few-shot
Task에 대한 few examples(1개에서 64개까지)을 제공한다.

모델은 해당 examples을 input으로 받고 answer 또는 different options에 대해 ranks를 매기는 방식을 사용한다.

<br>

## Instruction Finetuning

<br>

MMLU(Massive Multitask Language Understanding)에서 아주 적은 finetuning으로 performance 개선을 이끌어냈다.

해당 finetuning 모델은 LLaMA-I로 LLaMA-65B 모델을 protocol & instruction dataset으로 finetuning 한 것이다.

Instruction model인 OPT-IML, Flan-PaLM과 비교해 성능이 우세하지만, GPT3.5 + Optimized for code-completion tasks를 뛰어넘지는 못했다.

결과는 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/4e10279d-c17f-4c06-b4b4-befc3b80dc7d" width='400' height='400'>

</p> 

<br>

## Bias, Toxicity and Misinformation

<br>

해당 섹션에서는 Hallucination과 Bias 그리고 Toxicity한 Generation 정도를 판단하기 위해 여러 실험 결과에 대해 설명하고 있다.

해당 파트도 결과 위주이기에 논문을 참고하도록 하자.

<br>

## Conclusion

<br>

제안한 모델인 LLaMA는 SOTA foundation models이며, 가장 주목할 점은 LLaMA-13B가 GPT-3보다 더욱 나은 성능을 보이고 모델 크기는 10배보다 작다는 것이다.

또한 LLaMA-65B는 Chinchilla-70B와 PaLM-540B와 경쟁력이 있는 성능을 보인다.

마지막으로 해당 논문에서는 기존의 연구들과 달리, 소유권이 존재하는 데이터와 접근 불가한 데이터 없이 오로지 공공의 데이터만을 사용해서 SOTA를 달성할 수 있다는 것을 보여준데에 의의가 있다.

<br>