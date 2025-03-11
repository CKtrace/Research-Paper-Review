# [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860) (ICLR 2019)

<br>

## Abstract

<br>

Transformer는 longer-term dependency 특징을 지니고 있지만, fixed-length context의 한계가 존재한다.

본 논문에서 제안하는 Transfomer-XL은 longer-term dependency뿐만 지니는 것이 아니라, LM이 단어 예측시 사용 가능한 정보의 결핍 문제(Context fragmentation problem)까지 해결할 수 있다.

Transfomer-XL은 RNN보다 80%, Vanilla Transformer보다 450% longer-term dependency를 가질 뿐만 아니라, short & long sequences에서 모두 더 좋은 성능을 거둔다.

또한 evaluation 과정에서 Vanilla Transformer보다 1,800+ 빠르다.

<br>

## Introduction

<br>

LM에서는 long-term dependency를 갖는 것이 중요하고, 이를 위한 연구들이 활발히 진행되었다.

RNN과 LSTM은 standard한 solution이지만, RNN은 gradient vanishing and explosion 문제로 인해 최적화하기 힘들고, LSTM을 이용한 방법은 해당 문제를 충분히 해결하지는 못했다고 한다.

추후에 등장한 Transformer은 Context fragmentation problem을 겪었다.

이러한 문제점을 해결하고자 Transformer-XL을 제안한다.

Transformer-XL은 Recurrent connections을 통해 정보를 propagate함으로써 very long-term dependency를 갖는다.

Transformer-XL은 absolute positional encoding 대신 relative positional encoding을 사용하여 temporal confusion 없이 state를 reuse할 수 있게 했다.

또한 Transformer-XL은 RNN의 chracter-level과 word-level의 language modeling 성능을 넘어선 첫번째 self-attention model이다.

<br>

## Related Work

<br>

과거부터 지금까지 long-term dependency를 위한 여러 연구와 노력들이 있었다.

본 연구도 Long-term dependency를 해결하고자 했으며, 기존의 LSTM 구조와 RNN 구조를 사용한 연구들과 달리 transformer 구조를 base로 삼고 longer-term dependency Laerning을 통한 real-world task에서의 이점을 가져온다.

<br>

## Model

<br>

### Vanilla Transformer Language Models

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/2e48f068-42b2-4921-8dc1-8e1184f4a301" width='700' height='250'>

</p> 

<br>

LM에 Transformer나 self-attention을 적용할 때, 어떻게 학습해야 Transformer를 임의 long context를 고정된 크기의 representation으로 효율적인 encoding을 할지에 대한 central problem이 존재한다.

무한한 memory와 computation이 주어진다면, 가장 간단한 방법은 모든 context sequence를 unconditional Transformer decoder로 processing하는 것인데 이는 feed-forward neural network와 동일하다.

그러나, 이는 제한된 resource를 가진 학습 과정에서 거의 실행 불가하다.

이러한 문제 때문에 적당한 사이즈로 Segmentation을 해서 학습을 진행하면, 정보가 흐르지 않는다는 단점이 존재한다.

해당 방법을 Vanilla Transformer라고 하며 두 가지 중요한 문제점을 지니고 있다.

1. Segment 길이가 문장에 따라 달라질 가능성이 커서, 수 백가지 가능성이 생긴다.
2. Padding을 적용해 최대 길이로 활용 가능하나 너무 비효율적이다.

Evaluation 동안, 마지막 단어를 한 개씩 예측하기 위해 training과 동일한 길이의 segment를 소비하는데 이는 extremely expensive하다.

<br>

### Segment-Level Recurrence with State Reuse

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/d0c6c38d-4d9a-4efb-ad84-91e8d051832b" width='700' height='250'>

</p> 

<br>

위에서 언급한 fixed-length를 사용할 때의 문제점을 해결하고자 해당 연구에서는 Transformer 구조에 recurrence mechanism을 적용한다.

