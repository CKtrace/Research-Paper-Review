# [Going deeper with Convolutions](https://arxiv.org/abs/1409.4842)

## Abstract

본 연구에서는 'Inception'이라는 Deep Convolutional Neural Network Architecture를 제안했다. 

주요 달성사항은 제안한 Architecture가 Network 연산 자원 활용성을 개선시킨 사항이다. 

섬세하게 가공한 설계 덕분에 주어진 연산량을 지키면서 Network의 깊이와 폭을 증가시킬 수 있었다.

품질 최적화를 위한 모델 구성 결정은 Hebbian Principle과 Multi-Scale Processing에 대한 직관을 기반으로 하였다.

Inception을 이용하여 제안한 22 Layer를 지닌 모델인 GoogLeNet은 ILSVRC14의 Classification 부문과 Detection 부문에서 우수한 성적을 거두었다.

<br>
<br>

## Introduction
본 연구에서 제안한 모델인 GoogLeNet은 AlexNet과 비교하여 ILSVRC 2014에서 12배 더 적은 파라미터 수와 더 좋은 결과를 도출했다고 한다.

Object-Detection에서의 가장 큰 성과는 하나의 Deep Network나 큰 모델에서 나온 것이 아닌, R-CNN과 같은 Deep Architecture과 Classical Computer Vision의 시너지에서 나왔다는 것이다. 

또 다른 주목할만 한 성과는 Mobile과 Embedded Computing이 발전하는 가운데, 그들이 제안한 모델의 효율성(특히, 전력과 메모리 사용량)은 중요성을 갖는다.

제안한 모델은 규모가 학문적 호기심에 그치지 않고, 실세계의 큰 데이터셋에도 불구하고 주어진 연산량을 유지하며 합리적인 학습을 할 수 있도록 설계하였다고 한다.


<br>
<br>

## Related Notable Work

#### LeNet-5
LeNet-5의 Stack 된 Convolutional Layers와 한 개 또는 여러 개의 Fully Connected Layer를 갖는 구조는 CNN의 기본적인 구조이다.

해당 모델은 여러 데이터에서 우수한 성적을 보여주었으며, Imagenet과 같은 큰 데이터셋에 대해서, 최근 트렌드는 과적합을 막기 위한 Dropout 기법을 사용하면서 Layer의 개수와 사이즈를 늘리는것이다.

<br>

#### 1x1 Convolution

1x1 Convolution을 사용하는 데이터는 두 가지 목적이 존재한다.
```
1. Computational Bottlenecks을 제거하기 위한 차원 축소

2. Networks의 깊이와 폭을 Critical한 페널티 없이 증가
```

<br>
<br>

## Motivation and High Level Considerations

Deep Neural Network의 성능 개선을 위한 간단한 방법은 Network의 사이즈를 키우는 것이다. 

하지만 이 간단한 해결 방법은 두 가지 단점을 갖는다.

```
1. 모델의 사이즈가 크다는 것은 파라미터 수가 많다는 것이고, 이는 과적합되기 쉽다.

2. 획일하게 모델 사이즈를 키우면 자원 연산량이 급격하게 불어난다.
```

<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/29efaa11-6be8-4f7a-a849-3943e799211b">

</p> 

<br>

위 두 가지 단점을 해결하는 근본적인 방법은 내부에 Convolution이 있더라도 Fully Connected한 구조를 Sparsely Connected한 구조로 바꾸는 것이다.

<br>
<br>

## Architecural Networks

본 연구에서는 Inception Module이 서로의 위에 쌓이면서 그들의 장점이 부각될 것이라 한다.

허나 간단하게 이를 구조화했을 때, 5x5 Convolution의 수가 많지 않더라도 이 Convolution은 수 많은 필터를 갖는 Convolution들 위에 쌓기에는 상당히 많은 연산량을 요구하게 될 것이다.

그래서 본 논문에서 제안하는 방법은 바로 'Naive Inception Module' 구조에서 1x1 Convolution을 추가하는 것이다. 

<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/6015b3fc-26fb-4512-b308-7fd3fd411a6b">

</p> 

<br>

1x1 Convolution은 연산량이 비교적 많이 요구되는 3x3, 5x5 Convolution 전에 부착하여 연산량을 감소시키는데 사용한다.

추가적으로 ReLU 또한 연산량을 감소시키기 위해 사용한다.

Training 동안의 메모리 효율성을 위해 하위 Layer에서는 기존의 Convolution 형태를 가져가고, 상위 Layer에서는 Inception 모듈을 사용하였다고 한다.

Inception의 중요 이점 중 하나는 각 층에서 유닛들의 개수를 늘려도 연산 복잡도가 제어할 수 없을 정도로 불어나지 않는다는 것이다.

<br>
<br>

## GoogLeNet

<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/adbbc1d5-ea49-403c-ae85-9cdacab6d5c3">

</p> 

<br>


_본 파트에서는 GoogLeNet의 Architecture를 특징에 맞게 나누어서 설명하도록 하겠다._ 

<br>

### Part A
<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/0a0ac696-7f6e-4196-a7a7-cb4c64a52601">

</p> 

<br>

해당 부분은 입력과 가장 가까운 Layer들이 있는 곳이다.

_Architecture Networks_ 에서 설명한 것과 같이 메모리 효율성을 위해 낮은 Layer에서는 기본적인 CNN 구조를 띄고 있는 것을 확인할 수 있다.

<br>

### Part B
<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/f4bdba7c-7575-4e54-ba58-be5d27161dd7">

</p> 

<br>

해당 부분은 Inception Module이며, 다양한 Feature를 추출하기 위해 1x1 Convolution과 3x3 Convolution, 5x5 Convolution이 병렬적으로 연산을 수행하는 구조를 띄고 있다.

또한, 차원 축소를 위한 1x1 Convolution Layer가 탑재되어 있는 것을 확인할 수 있다.

<br>

### Part C
<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/df15e9d0-6ac6-4669-9cd1-4a5dc7fa1624">

</p> 

<br>

해당 부분은 Auxiliary Classifier가 적용된 부분이다.

모델이 깊으면 깊을수록, Gradient Vanishing 문제를 갖기 쉬워진다.

이때,  Auxiliary Classifier들을 Middle Layer에 추가함으로써, 저수준에서의 분류를 더 원활하게 할 수 있으며 역전파 되는 경사 신호를 증폭시켜 Saturated 되지 않게 할 뿐만 아니라 추가적인 Regularization을 제공 가능하다.

또한 Inference할 때는 이 분류기들을 모두 제거한다고 한다.

<br>

### Part D
<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/6268cf7e-192d-42ee-ac2d-a464083b8021">

</p> 

<br>

Part D는 GoogLeNet의 끝 부분이다.

해당 부분에서 최종 Classifier 이전에 Average Pooling Layer를 사용하고 있으며 이는 Golbal Average Pooling(GAP)가 적용되었다.

GAP는 이전 Layer에서 추출된 Feature Map들을 각각 평균을 낸 후 그것들을 이어 1차원 벡터로 만든다.

1차원 벡터로 만든 값을 최종 이미지 분류를 위한 Softmax Layer와 연결한다.


GAP를 사용하여 1차원 벡터로 만들면 가중치의 개수를 상당히 많이 줄여준다고 한다.


기존의 Fully Connected 방식을 이용할 경우에는 Weight의 개수가 7 x 7 x 1024 x 1024 = 51.3M이지만, GAP를 사용하면 단 1개의 가중치도 필요하지 않다.

왜냐하면, 평균을 내는 것이기 때문이다.

또한 GAP를 적용할 시, fine tuning을 하기 쉽게 만든다고 한다.


마지막으로 하단의 이미지는 GoogLeNet의 전체 Structure이다.

<br>

<p align="center">

  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/cfd90801-cd76-408b-b431-5e2d5db3ec35">

</p> 

<br>

<br>


## Training Methodology

- SGD with 0.9 Momemtum
- Learning Rate 8 Epoch마다 4%씩 감소
- 이미지의 비율을 3:4, 4:3으로 유지
- 다양한 크기의 Patch 사용으로 Original Size의 8% ~ 100% 포함
- Photometric Distortions 증강 기법을 사용하여 학습 데이터 늘림


<br>

<br>

## Conclusion

Inception은 Sparse한 구조를 Dense한 구조로 근사화 시켜 성능 개선을 이끌었다.

해당 구조는 기존의 CNN의 성능을 개선시키는 새로운 방법이었으며, 약간의 연산량 증가로 성능을 대폭 상승하는 결과를 도출했다.

<br>

<br>
