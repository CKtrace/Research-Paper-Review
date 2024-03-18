# [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

<br>

## Abstract

본 연구에서는 이전에 사용하던 것들보다 더 깊고 학습에 용이한 residual learning framework를 제안한다.

Residual Networks는 Optimize하기 쉬우며 상당한 깊이 증가로도 우수한 정확도를 도출한다.

ImageNet Dataset으로 평가한 Resnet은 152Layers를 갖고 있으며, 이는 VGGNet에 비해 8배 더 깊으면서 복잡도는 더 낮았다고 한다.

그 뿐만 아니라 Object Detection 부문에서도 좋은 성과를 냈다고 한다.

<br>
<br>

## Introduction

본 파트에서는 Deeper Network를 만들 때 Vanishing Problem과 Exploding Problem에 직면한다고 한다.

SGD를 적용한 10개의 layer까지는 Normalization 기법과 Batch Normalization과 같은 Intermediate Normalization Layer를 사용했을 경우, 문제가 없었다고 한다.

허나 Network가 더욱 깊어질수록 해당 효과는 미비해진다.

Deeper Network가 수렴할 때, Degradation Problem이 발생하는데, 이는 Network의 깊이가 깊어질수록 Accuracy는 포화(saturation)되고, 하락(Degradation)하는 문제이다.

이러한 Degradation Problem은 Overfitting 때문이 아니며 layer들을 추가하여 만든 Deep Network는 높은 학습 오류를 기록하는 것을 실험을 통해 알 수 있었다고 한다.

해당 문제를 현존하는 해결 방법인 Identity Mapping을 사용하면, Deep Model을 사용하여도 높은 학습 오류를 기록하지는 않는다.

<br>

<figure class="half">  
<a href="link"><img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/f73cf7a7-8512-4e05-95dc-f333943ff046"></a> 
<a href="link"><img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/c17f38c6-59c6-4e33-9fd1-653b7c1fe344"></a>  
<br>
<figcaption> Underlying Mapping(Unreferenced Mapping) & Identity Mapping(Residual Mapping) </figcaption> </figure>

<br>
<br>

본 연구에서는 Degradation Problem을 피할 수 있는 Deep Residual Learning Framework를 소개한다.

몇몇 Stack된 Layer들에게 Underlying Mapping이 아닌 Residual Mapping을 사용한다.

기존 Normal Network의 Underlying Mapping의 수식이 H(x)라면, 즉 H(x)를 최소시키는 것이 학습의 목표이다.

해당 논문의 목표는 F(X) = H(x) - x를 최소화시키는 Residual Mapping으로 H(x)를 재정의한다.

그들이 세운 가설은 Residual Mapping이 기존의 unreferenced Mapping보다 쉽게 Optimize될 것이라는 거다.

Skip Connection을 이용하여 Indentity Mapping 구조를 만들어주고, 추가적인 Parameter나 Computational Complexity는 없다.

H(x)를 0에 근사시키면 Gradient Vanishing Problem이 발생하는 반면, F(x)는 0 + 1이 되며 Gradient Vanishing Problem을 피할 수 있게 되며 이는 High Accuracy를 갖는 Deeper Network를 만들 수 있게 한다.


<br>
<br>

## Related Work

### 1. Residual Representations

Vetor Quantization에 있어, Residual Vector를 Encoding하는 것이 Original Vector보다 더욱 효과적이며, 

이때 Vetor Quantization이란, Feature Vector X를 Class Vector Y로 Mapping 하는 것을 뜻한다.

<br>

### 2. Shortcut Connections

Rsenet의 Identity Mapping의 핵심인 Skip Connections은 추가적인 Parmeter가 없고 Gradient Vanishing Problem으로부터 자유로워 모든 정보가 통과할 수 있다고 한다.

이로 인해, Residual Function(F(x))의 지속 학습이 가능하다.


<br>
<br>


## Deep Residual Learning

앞서 서술했듯이 Shortcut Connection은 추가적인 Parameter나 Computational Complexity를 필요로 하지 않는다.

또한, Indentity Mapping에서 x와 F는 차원이 같아야 하는데, 이들이 다르다면 Linear Projection W_s를 곱하여 차원을 같게 해준다.


<br>
<br>

## Network Architectures

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/082f256d-7525-48a0-af3c-cc814ed9b095">
</p> 

<br>


### Plain Network

Plain Network는 VGGNet의 철학에 영감을 받아 제작되었다고 한다.

해당 Network의 Layer는 34개이며, 주목해야 할 점은 VGGNet보다 적은 Filters와 적은 Complexity를 갖고 있다고 한다.

<br>

### Residual Network

Residual Network는 Plain NetworK에 Shortcut Connections을 추가한 구조이다.

Identity Shortcuts은 입출력의 차원이 같은 경우에 사용할 수 있다.

만약 입출력의 차원이 다른 경우 선택할 수 있는 두 옵션은 아래와 같다.

```
1. Zero Padding 기법을 이용하여 차원의 수를 일치시키고 Indetity Shortcut 사용

2. Linear Projection W_s를 곱하여 차원을 같게 해주는 방법(1x1 Convolution)
```


<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/bc20f9a1-6afb-49d1-89d9-ea7fc1b3bf6f">
</p> 

<br>

<br>
<br>


## Implementation

1. 짧은 쪽이 [256, 480] 사이가 되도록 Random 하게 Resize 수행

2. Horizontal flip 부분적으로 적용 및 Per-Pixel Mean을 빼준다

3. 224 x 224 사이즈로 Random 하게 Crop 수행

4. Standard Color Augmentation 적용

5. z에 Batch Normalization 적용

6. He 초기화 방법으로 가중치 초기화

7. Optimizer : SGD (Mini-Batch Size = 256) 

8. Learning rate : 0.1에서 시작 (Training이 정체될 때 10씩 나눠준다)

9. Weight decay : 0.0001

10. Momentum : 0.9

11. 60 X 10^4 반복 수행

12. Dropout 사용 X

<br>


Test에서는 10-Cross Validation 방식을 적용한다.

또한 Multiple Scale을 적용해 짧은 쪽이 {224, 256, 384, 480, 640} 중 하나가 되도록 Resize 한 후, 평균 Score를 산출한다.

<BR>
<BR>

## Experiment

### ImageNet Classification

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/1d026c8d-51e5-4220-a98c-536321cdd489">
</p> 

<br>

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/f742007d-6d49-415b-b39b-a78d7b03ea6b">
</p> 

<br>

#### Plain Network

해당 Plain Network는 Batch Normalization을 적용하여 학습함을 통해 Forward Propagated Signal이 0이 되지 않도록 할 수 있었다.

또한 Backward Propagated Gradients 또한 healthy norm을 보여준다.

즉, Forward, Backward signal Vanishing Problem이 아닌 Deep Plain Networks가 Exponentially Low convergence Rate를 갖고 있어 training error를 줄인 것으로 예측할 수 있었다고 한다.

<br>

#### Residual Network
 
그들은 모든 Shortcut에 Identity Mapping과 차원 확대를 위한 zero-padding이 적용하여 Plain Network와 Parameter 수 차이가 없는 Residual Network를 실험에 사용하였다.

깊이를 늘려도 Degradation Problem을 피할 수 있었으며, 실험 결과를 통해서 이를 알 수 있었다.

또한 ResNet은 수렴 속도 또한 빠른 것을 알 수 있었다.

<br>
<br>

## Identity _vs_ Projection Shortcut

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/caa0b724-5fb6-43f0-ab21-592c85a5cc11">
</p> 

<br>

해당 파트에서는 Identity Shortcut과 Projection Shortcut과의 비교한다.

아래 3가지 (A-C) 옵션을 통해 Projection Shortcut의 필요성에 대해 알아보겠다.

(A) 차원을 맞추기 위한 Zero-Padding Shortcut(모든 Shortcut은 Parameter-Free하다.)
(B) Projection Shortcut은 차원을 늘리기 위해서만 사용하며, 이외의 Shortcuts은 모두 Identity Shortcuts
(C) 모든 Shortcuts은 Projection Shortcuts을 사용

3가지 옵션을 각각 적용한 ResNet의 결과표를 보았을 때, 세 가지 결과 모두 큰 차이가 없는 것을 알 수 있었다.

이를 통해 Degradation Problem을 해결하기 위해서는 Projection Shortcuts이 반드시 필요한 것은 아닌 것을 알 수 있었다.

또한 Projection Shortcuts은 Computation Complexity를 늘린다.

결과적으로 이제는 (C) 옵션을 사용하지 않는다고 한다.

<br>
<br>

## Deeper Bottlenexk Architectures

<br>

<p align="center">
  <img src='https://github.com/CKtrace/Research-Paper-Review/assets/97859215/078e93ea-1c25-442f-a4ad-dd93aa1bd1e5'>
</p> 

<br>

그들은 Fig5의 Building Block을 수정하여 Bottleneck Design을 제작하였다.

Bottleneck Design은 1x1, 3x3 그리고 1x1 Convolutions으로 이루어져 있다.

1x1 Convolutions은 차원 축소 및 확대 역할을 하며, 3x3 Convolutions은 축소된 입출력 차원의 bottleneck이다.

만약 Identity Shortcut 대신 Projection Shortcut이 대체한다면, Shortcut이 두 고차원에 연결되어 시간 복잡도와 Model Size는 두 배 커지는 것을 볼 수 있다.

따라서 Identity Shortcut은 Bottleneck Design을 한 모델을 더욱 효율적이게 만들어준다.

놀랍게도, 152 Layer를 갖는 ResNet은 VGG-16/19 Networks보다 Lower Complexity를 갖는 것을 알 수 있으며, 152 Layer까지도 34 Layer를 갖는 ResNet보다 더욱 정확한 것을 확인할 수 있었다.

정리하자면, Bottleneck Design을 통해 Degradation Problem이 존재하지 않아 상당한 깊이의 증가에도 상당한 정확도를 얻을 수 있었다는 것이다.

<br>
<br>

## Exploring Over 1000 Layers

실험 결과에 의하면 1202-Layer ResNet과 110-Layer ResNet은 유사한 Training Error를 갖지만, 1202-Layer ResNet은 110-Layer ResNet보다 결과가 좋지 않았다.

연구진들의 의논 결과 그 이유는 Overfitting 때문이라 생각한다고 한다.

왜냐하면 Strong Regularization(Such as Maxout, Dropout..)이 적용되지 않았기 때문에 Overfitting은 일어났을 것이라고 한다.

추후 연구에서는 Strong Regularization과 결합하여 개선된 결과를 보여줄 것이라 한다.


<br>
<br>
