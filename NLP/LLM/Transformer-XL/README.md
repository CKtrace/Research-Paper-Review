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

Transformer-XL은 fixed & cached한 previous segment를 next new segment에 재사용한다.

이러한 Additional Input은 과거 정보를 충분하게 하고, 이는 longer-term dependency를 가지며 context fragmentation(정보 불충분)을 피하게 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a1c460f0-c5da-4083-af60-42c71ecf436c" width='300' height='50'>

</p> 

<br>

위의 값은 new segment의 n-1 hidden layer의 할당되는 값이며, 실제로 previous segment의 동일한 위치의 hidden layer의 stop-gradient를 한 값을 현재 값과 concat하는 것을 확인할 수 있다.

Stop-gradient를 통해 backpropagation시, previous segment의 hidden state에서 사용된 parameter는 학습되지 않는다.

W는 모델의 파라미터를 의미한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/833855fe-8e5b-417c-bb56-956a202d3b86" width='700' height='50'>

</p> 

<br>

Transformer-XL의 q, k, v의 값은 위와 같으며, 기존의 transformer와 달리 key, value에 이전 segment의 정보가 들어있음을 알 수 있다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/579e7436-837d-466f-b2a0-a072206008a0" width='500' height='50'>

</p> 

<br>

만들어진 Query, Key, Value를 이용해 현재 Segment의 n번 째 hidden layer의 hidden state를 만든다.


하지만 수식을 보면 n일 때의 q, k, v를 n-1일 때의 값을 이용해 구하기 때문에, Segment 하나 당 하나의 layer씩 밀린다.

이러한 문제점으로 인해 최대 의존 관계는 (Segment Length x Layer 개수)로 제한했다.

위 fig2의 (b)를 보면 Segment Length가 4이고, Layer 개수가 3이므로 최대 12의 의존성을 가질 수 있다.

Evaluation 시, vanilla Transformer처럼 sliding windows 방식이 아닌 previous segment를 cahced 해서 사용하기 때문에 약 1,800 배 빠른 연산이 가능하다.


<br>

### Relative Positional Encodings

<br>

위에서 말한 방법을 사용하기 위해서는 해결해야 하는 문제가 존재한다.

바로 Positional Encoding이다.

Vanilla Transformer는 Word embedding 값에 positional encoding(absolute Position) 값을 더해주는 방식을 사용하는데, 이는 이전 state와 현재 state를 구성할 때 사용하는 positional encoding 값이 같게 되어 제안하는 Transformer-XL에서는 사용할 수 없다.

Positional Encoding은 Model에게 토큰의 위치에 대한 Clue/Bias를 제공하는데, 이를 절대적인 방식이 아닌 상대적인 방식으로 제공하고자 Relative Positional Encoding 방식을 제안한다.

해당 방식은 각 토큰의 위치를 나타내는 absolute Encoding 대신 두 토큰 사이의 거리를 나타내는 0 ~ L_max 사이의 Relative Encoding을 만드는 방식이다.

위에서 언급한 Vanilla Transformer의 차이점을 제외한 그 이외의 구성은 동일하다.

<br>

## Experiment

<br>

본 논문에서는 Transformer-XL를 여러 데이터 셋 (WikiText-103, enwik8, text8, One Billion Word)으로 실험을 해 비교를 했으며 각 데이터 셋을 통해 보여주고 싶었던 내용은 아래와 같다.

1. WikiText-103 => 가장 큰 Word 단위를 가지며, long-term Dependency를 측정
2. enwik8 => 정제되지 않은 Wikipedia 데이터셋에서의 성능
3. text8 => 정제된 Wikipedia 데이터셋에서의 성능
4. One Billion Word => 문장들을 섞어 Long-term dependency 보존 X, Short-term dependency 측정

<br>

### 1. WikiText-103


<p align="center">

  <img src="https://github.com/user-attachments/assets/30c31e04-71f6-4138-ad87-abff8e9b9bfc" width='500' height='350'>

</p> 

<br>

### 2. enwik8


<p align="center">

  <img src="https://github.com/user-attachments/assets/2e0ea67a-4534-4133-899c-7c270a58ecf6" width='500' height='350'>

</p> 

<br>

Transformer-XL은 Layer가 많아질수록 더욱 좋은 성능을 보이는 것을 확인할 수 있었다.

<br>

### 3. text8


<p align="center">

  <img src="https://github.com/user-attachments/assets/4f9f55a6-3426-485e-b7c3-18e9dc3ea204" width='500' height='350'>

</p> 

<br>

본 실험에서는 enwik8에서 좋은 성능을 보여준 parameter로 학습을 진행했고 SOTA 달성

<br>

### 4. One Billion Word

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/92a74c72-b5c4-4367-8a0c-025d3262fc17" width='500' height='350'>

</p> 

<br>

본 실험에서는 Transformer-XL이 short-term dependency에도 Robust하다는 것을 보여준 실험이었다.

<br>

## Conclusion

<br>

Transformer-XL은 perplexity에 강한 결과들을 보여주었으며, RNN, LSTM보다 Long-term dependency한 성격을 보여주었다.

그 뿐만 아니라, evaluation 속도 또한 증가시킬 수 있었다.


