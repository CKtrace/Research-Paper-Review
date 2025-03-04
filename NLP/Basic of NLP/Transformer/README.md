# Attention Is All You Need (NIPS 2017)

<br>

## Abstract

<br>

Sequence transduction model은 encoder와 decoder를 포함한 recurrent 또는 CNN에 기초한다.

해당 논문에서 제안하는 모델인 Transformer는 오로지 Atttention mechanism에만 기초를 둔다.

Transformer는 크고 제한된 데이터에서의 영어 구문 분석 성능이 좋아 여러 tasks에서 잘 일반화가 된다고 한다.

<br>

## Introduction

<br>

RNN, LSTM, Gated RNN는 sequence modelingrhk transduction problem에서 좋은 성능을 보인다.

하지만 위와 같은 Recurrent Model은 이전의 데이터를 가져와 추가로 input에 넣는 방식이기 때문에, sequence 길이는 직접적으로 제한된 메모리에 영향을 미치는 치명적인 문제가 존재한다.

이후 여러 연구에서 해당 문제를 해결하기 위해 많은 노력이 존재했지만 여전히 그 문제는 남아있다.

Attention Mechanism은 input과 output sequence의 distance를 가져갈 수 있게 한다.

```
Attention Mechanism Figure
Fig Source : https://wikidocs.net/22893
```
<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/78fc7121-5516-49d8-a585-bbd7a40562f0" width='700' height='350'>

</p> 

<br>

해당 논문에서 제안하는 Transformer는 Recurrence를 삼가고, 오로지 Attention Mechanism만을 기반으로 구조가 이뤄진다.

Transformer는 더욱 병렬에 적합하고 번역에서 SOTA를 달성했다.

<br>

## Model Architecture

<br>

성능이 좋은 대다수의 Neural Sequence transfuction model은 encoder-decoder 구조를 띈다.

Transformer는 encoder와 decoder 모두에서 쌓은 self-attention과 point-wise FC layer를 사용한다.


<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a204e464-e1c8-4afb-a04b-63e3b7dad65d" width='350' height='500'>

</p> 

<br>

### Encoder and Decoder Stacks

<br>

#### Encoder

<br>

Encoder는 N=6 개의 동일한 layer로 구성되며, 각 layer는 아래 두 개의 sub-layer로 구성된다.

```
- Multi-Head self-attention
- Position-wise FC Feed-Forward Network
```

각 sub-layer의 출력은 LayerNorm(x + Sublayer(x))이며, Sublayer(x)는 sub-layer 자체로 구현되는 함수이다.

Residual Connection을 용이하게 하기 위해, Embedding layer를 포함한 모든 sub-layer의 output dimension은 512이다.

<br>

#### Decoder

<br>

Decoder는 N=6개의 Encoder와 동일한 레이어로 구성되지만 한 가지 sub-layer를 더 가진다.

```
- Encoder Stack 출력값에 multi-head attention을 수행하는 sub-layer
```

Encoder와 유사하게 residual connection이 각 sub-layer의 정규화 layer 뒤에 존재하고 Decoder가 출력을 생성 할 때, 다음 출력에서 정보를 얻는 것을 방지하기 위해 Masking 기법이 적용된 Masked Multi-Head Attention을 사용한다.

해당 방법은 i번 째 원소 생서 시에 1 ~ i-1번 째 원소만 참조하도록 한다.


<br>

### Attention

<br>

Attention Function은 query와 key-value로 output을 도출한다.

query, key, value, output은 모두 벡터 값이다.

Output은 vlaue들의 가중합으로 계산되며, 각 value의 가중치는 query와 연관된 key의 Compatibility function에 의해 계산된다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/2636eaa4-b282-473a-9663-c276f805340d" width='500' height='300'>

</p> 

<br>

### Scaled Dot-product Attention

<br>

Scaled Dot-Product Attention의 input 값은 query와 d_k 차원을 갖는 key, d_v 차원을 갖는 value가 들어간다.

query와 모든 key 값들은 내적을 하고 그 값을 root(d_k)로 나눈 뒤, softmax function을 태워 value들의 가중치를 구한다.

실제로는 query들을 동시에 계산하기 위해 행렬 Q로 묶고, key, value도 행렬 K, V로 표현한다.

```
Attention(Q, K, V) = softmax(QK^T/root(d_k))V
```

Attention Function은 additive function, dot-product attention에서 많이 쓰이는데, dot-product attention은 root(d_k)로 나누는 식을 제외하고는 동일하다.

두 함수는 복잡도는 이론상 유사하지만, 실제론 dot-product attention은 higly optimized matrix multiplication code를 적용 가능하기에 훨씬 빠르고 공간 효율성이 좋다.

d_k의 값이 작으면, 두 attention mechanism은 결과가 유사하게 나오지만, d_k의 값이 크면, 내적값이 커지고 이 값을 softmax에 넣으면 vanishing gradient problme을 겪게 된다.

이러한 이유로 root(d_k) 값을 곱했다고 한다.

<br>

### Multi-Head Attention

<br>

Single Attention Function(Scaled Dot-Product Attention)을 사용하는 것보다 각각 d_k, d_k, d_v 차원을 갖는 query, key, value을 다르게 h번 학습시키는 것이 낫다고 한다.

이때, h번 학습한다는 의미는 각 sub-layer에 동일한 부분이 h개 존재한다는 것이다.

각각 따로 계산된 h쌍의 d_v 차원의 출력은 Concat 후 한 번 더 Linear Function을 통과 시켜 최종 출력값을 얻는다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/70d5cdf5-28c5-488d-b225-7f790bba8e87" width='500' height='100'>

</p> 

<br>

해당 수식을 이용하기에, 각 head에서의 차원이 줄어 사실상 Single Attention Function 계산량과 유사하다.

<br>

### Application of Attention in our Model

<br>

Transformer는 multi-head attention을 아래와 같은 세 가지 방법으로 사용한다.
(Transformer figure랑 같이 보면 이해하기 더 좋음)

```
1. 'Encoder-Decoder attention' layer에서 query는 이전 Decoder에서 오며 memory keys와 values는 encoder의 output에서 온다. 이러한 방법은 Decoder가 입력의 모든 원소를 고려할 수 있게 한다.

2. Encoder는 self-attention layer를 포함한다. Self-attention later의 모든 key, value, query는 같은 곳에서 오며 이는 encoder 이전 layer의 출력에서 온다. 따라서 Encoder의 각 원소는 이전 layer의 모든 원소를 고려 가능하다.

3. Decoder도 이와 유사하지만, auto-regressive의 속성 보존을 위해 Decoder 출력 생성 시 다음 출력을 고려해서는 안된다. 이 때문에 Masking을 통해 이전 원소는 참조하지 않는다. 해당 Masking은 dot-product 수행 시 -nan으로 설정함으로써 masking out시키고, 이 값이 softmax를 통과 시 0이 된다.
```

<br>

### Position-wise Feed-Forward Networks

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/554c1016-ba0d-475c-a87c-d81eae6c8677" width='500' height='70'>

</p> 

<br>

Encoder와 Decoder의 각 Layer는 FC feed-forward 네트워크를 포함하며, 각 위치마다 동일 적용되지만 각각 따로 적용된다. RELU 함수와 2개의 선형 변환을 포함한다.

각 Layer의 해당 부분은 독립적인 parameter를 사용한다.

<br>

### Embedding and Softmax

<br>

타 모델과 유사하게, 학습된 embedding을 통해 d_model 차원의 벡터로 input 토큰과 output 토큰을 변환했다.

또한 학습된 선형 변환과 softmax function으로 다음 토큰의 확률을 예측할 수 있도록 decoder의 output을 변환한다.

Transformer에서는 두 embedding layer와 pre-softmax 선형변환 사이에 같은 weight 행렬을 사용했고, Embedding layer에는 root(d_model)을 곱한다.

<br>

### Positional Encoding

<br>

Transformer는 recurrence나 convolution이 없이 sequence의 순서를 이용해야 했다.

이러한 이유 때문에, 문장에서 원소들의 위치에 관한 정보를 주입해줘야 했다.

그러기 위해 사용된 것이 Positional Encoding이다.

Positional Encoding은 d_model과 동일한 차원을 갖게 해 두 값이 합칠 수 있게 했다.

문장에서 원소들의 위치에 관한 정보를 구하는 방법은 여러 가지가 있다.

아래 figure는 해당 연구에서 사용한 방법이다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/5a52be24-86e0-4987-ad89-970ae1f94339" width='500' height='150'>

</p> 

<br>

```
pos : position
i : Dimension
```

sin 함수와 cos 함수를 사용했는데, 최종적으로는 sin 함수를 택했다.

그 이유는 sin 함수를 사용 시, 학습 때보다 더 긴 Sequence를 만나도 추정이 가능했기 때문이라 한다.

<br>

## Why Self-Attention

<br>

Self-Attention을 쓰는 세 가지 이유를 서술하는데 정리하자면 아래와 같다.

```
1. Layer 당 전체 계산량이 적다

2. 병렬 계산 처리가 가능하다.

3. 장거리 의존성의 학습 속도, 능력에서 Recurrent, Convolution보다 뛰어나다.
```

<br>

## Training

<br>

|Parameter Setting Value|Remark|
|:---:|:---:|
|Batch Size|25000|
|Hardware|NVIDIA P100 GPUS * 8|
|Time|Base Model -> 100,000stpes(12시간) & Big Model -> 300,000steps(3.5일)|
|Optimizer|Adam|
|Regularization|Residual Dropout (P=0.1)|
|Warmup_steps|4,000|
|Label Smoothing|Smoothing value = 0.1|

<br>

## Results

<br>

### Machine Translation

<p align="center">

  <img src="https://github.com/user-attachments/assets/48c83ba2-8237-4e3d-bc76-acfcaef926b3" width='700' height='300'>

</p> 

<br>

### Model Variations

<p align="center">

  <img src="https://github.com/user-attachments/assets/8573eb5d-10a1-4a5b-9898-7b738e3c0f6c" width='700' height='600'>

</p> 

<br>

<br>

### English constituency Parsing

<p align="center">

  <img src="https://github.com/user-attachments/assets/1f805b4a-6ee0-4823-bdc1-48313a597a0d" width='700' height='300'>

</p> 

<br>

## Conclusion

<br>

Transformer는 recurrence와 convolution을 모두 제거한, 오로지 attention에만 의존하는 새로운 종류의 모델임에 의의가 있다.

해당 모델은 병렬화를 통한 학습 속도 개선과 동시에 SOTA 또한 달성하였다.

Transformer는 텍스트 뿐만 아니라 이미지, 오디오, 비디오 등에서도 효과적으로 사용할 수 있을 것이라고 말하며 마친다.

<br>