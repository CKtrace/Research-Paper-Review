# [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)

<br>

## Abstract

본 연구는 Large-Scale 이미지 인식에서 Convolutional Nerwork의 깊이가 주는 영향에 대해 설명한다.
<br>
3x3 Convolution Layer만 사용하여 Network의 깊이를 늘렸으며, 16-19개의 layer까지 늘리며 성능 개선에 성공하였다고 한다.

<br>
<br>

## Introduction

본 논문은 Convolution Network 구조에서의 깊이의 중요성 측면에 대해 서술한다.
<br>
연구진들은 Architecture의 여러 파라미터를 고치면서, 꾸준하게 더 많은 3x3 Convolution Filter를 사용하여 Network의 깊이를 늘려나갔다.

<br>
<br>

## Architecture

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/31d3a3f9-b68b-4d48-8ad3-4aa6f3f022e4">
</p> 

<br>

- Input image는 224x224로 고정된 사이즈의 RGB 이미지를 사용
- 전처리 과정은 오직 RGB 값의 평균을 Trainning Set의 각 픽셀에 빼준 작업 뿐
- Convolution Stride는 1로 고정하였으며, Padding 또한 1로 고정한다.
- Max-Pooling에서 2x2 Window와 Stride는 1로 설정하였으며, 모든 Convolution layer에 Max-Pooling 사용하는 것은 아님
- A-E 중 C에는 1x1 Convolution Filter(with ReLU) 탑재
- Stacking 된 Convolution Layer들 후에는 세 개의 Fully Connected Layer 존재
- 모든 Hidden Layer에는 ReLU 부착
- 본 연구에서 제안하는 모델 하나를 제외하고는 Local Response Normalization (LRN) 사용 (실험 결과 LRN 사용 시 메모리와 시간만 잡아 먹고, 성능 개선에는 용이하지 않았다고 한다.)


<br>
<br>

## Discussion

본 파트에서는 Single 7x7 Convolutional Layer 대신 three stack 3x3 Convolutional Layer를 사용하였을 때, 얻은 두 가지 이점에 대해 서술한다.

<br>

```
1. Single ReLU를 사용하는 대신 세 개의 ReLU를 사용하면 결정함수가 더 구체적이게 된다.
2. 파라미터 수를 줄일 수 있는데, If Channel == C,

    - Single 7x7 Convolutional Layer 사용 시 Param => 7**2(C**2) = 49C**2
    - Three Stack 3x3 Convolutional Layer 사용 시 Param => 3(3**2(C**2)) = 27C**2
    - 약 81% 더 줄일 수 있다.
```

<br>

C 모델에서의 1x1 Convolutional Filter는 타 연구에서 사용된 적이 있지만, 그들이 제안한 Network는 본인들이 제안한 모델들보다 얕았으며, Large-Scale ILSVRC 데이터셋을 사용하지도 않았다고 한다.
그리고 GoogLeNet과 비교하며, 자신들이 제안한 모델은 덜 복잡하고, 첫 번째 Layer, Feature Map에서의 Spatial Resolution이 더욱 과감하게 연산량을 감소시켰다고 한다.


<br>
<br>

## CLASSIFICATION FRAMEWORK

<br>

### Training

- Cost Function
  - Multinomial Logistic Regression = Cross Entropy
 
- Mini Batch
  - 256 Size
 
- Optimizer
  - Momemtum = 0.9
 
- Regularization
  - L2 = 5.10 <sup>-4<sup>
  - Dropout = 0.5 in FC1 & FC2
 
- Learning Rate
  - 10<sup>-2 </sup> (성능 개선 안될 시 10배 감소) => 학습률은 실험 간 총 세 번 감소, 학습의 경우 74epoch 이후 종료

<br>

잘못된 가중치 초기화는 기울기 불안정성으로 Deep Nets의 학습을 멎게하므로, Network 가중치 초기화는 매우 중요하다.<br>
이러한 문제점을 방지하기 위해 랜덤 초기화를 이용하기에 충분히 얕은 A 모델로 학습을 시작했다고 한다.<br>
그리고 더욱 깊은 아키텍쳐를 학습시킬 때, A 모델의 초반 4개의 Convolution Layer들과 마지막 세 개의 FC Layer들을 초기화 시켰으며 중간 Layer들을 랜덤 초기화 하였다고 한다.<br>
초기화 이전 Layer들의 Learning Rate는 감소시키지 않았으며, Learning Rate가 학습하는 동안 바뀌도록 하였다.<br>
랜덤 초기화를 위해, Zero Mean과 분산(0.01)을 이용한 Normal Distribution을 가중치로부터 샘플링했다.<br>
편차는 0으로 초기화였다고 한다. <br>
추가적으로 본 논문 제출 후 Glorot & Bengio(2010)에서 Pre-Traning 없이 가중치 랜덤 초기화가 가능하다는 것을 찾았다고 한다.

<br>
<br>

### Training Image Size

Training Scale을 S라 하고 해당 Training Image 224x224로 Crop시킨다.<br>
이때, S=224인 경우, 이미지의 전체 부분이 포함되겠지만, S>224인 경우 이미지의 작은 부분에 해당할 것이며, 작은 부분에 해당할 것이며, 작은 물체 또는 물체의 일부만 포함될 것이다.

<br>

S를 설정하기 위해 아래와 같은 두 개의 방법을 고안했다.

<br><br>

#### 1. Single Scale Training

본 실험에서 두 가지 고정된 Scale _(S=256 & S=384)_ 로  학습된 모델을 평가하였다. <br>
먼저 S=256을 이용하여 학습을 시켰으며, S=384인 경우 학습 속도를 올리기 위해 S=256으로 Pretrained 된 가중치로 초기화하고 더욱 작은 Learning Rate _(10<sup>-3</sup>)_ 로 초기화하였다고 한다.

<br><br>

#### 2. Multi Scale Training

각각의 학습 이미지를 개개인으로 특정한 범위인 [S_min, S_max] _(본 연구에서 S_min = 256, S_max = 512)_ 에서 랜덤하게 개별적으로 Rescale 시킨다.<br>
이렇게 모든 이미지들이 다른 사이즈로 학습이 되면 이점을 갖는다고 한다.<br>
단일 모델들이 넓은 범위인 Scale들을 인식하며 학습하는 것을 _Scale Jittering_이라고 한다.<br>
마지막으로 속도 때문에 Multi Scale Training을 할 땐 같은 설정 사항에서 S=384로 고정된 상태에서 사전 훈련된 단일 Scale 모델의 모든 데이터로 _Fine-Tunning<sup>_기존에 학습되어져 있는 모델을 기반으로 Architecture를 새로운 목적(나의 이미지 데이터에 맞게) 변형하고 이미 학습된 모델 Weights로부터 학습을 업데이트 하는 방법_
</sup>_ 해준다.<br>

<br>
<br>

### Testing _(본 파트에서 TEST==Validation)_

<br>

Training Image를 Rescale 하는 것처럼 VGG 모델을 Test할 때도 Rescale 해주었다고 한다.<br>
Q를 Validation Image _(논문에서는 Test Image라고 한다. Validation Set을 Test Set으로 사용하였다고 논문에 서술되어 있다.)_ 의 Scale이라 할 때, 각각의 S값마다 다른 Q를 적용 시에 VGG 모델의 성능이 좋아졌다고 한다.<br>
이 말은 즉슨, Validation Set을 이용해 평가와 학습하는 효과까지 같이 가져간다는 것이다.<br>
또한 Training 할 때의 구조와 Valdiation 수행 시의 모델 구조는 다르다.<br>
Validation에서는 FC Layer들을 Convolution 계층으로 변환한다.

<br>

```
FC1 -> 7x7 Convolution Layer
FC2 & FC3 -> 1x1 Convolution Layer

해당 구조를 Fully Convolutional Network라고 한다.

FC Layer와 1x1 Convolution Layer가 대치될 수 있는 이유는
두 구조 모두 이전 데이터의 모든 노드가 다음 레이어의 모든 노드에 연결되기 때문이다.
```

<br>

그리고 모델에 Uncropped Image를 적용하게 된다.<br>
하지만, 최종 Output Feature Map Size는 입력되는 Image Size에 따라 달라지며, 1x1 사이즈가 아닌 Output Feature Map을 Class Score Map이라 한다.<br>
If, Class Score Map Size가 7 x 7 x 1000인 경우 Mean or Average Pooling을 적용한다.<br>
그 후에 Softmax를 거치고 Flipped Image와 Original Image의 평균값을 통해 최종 Score를 출력한다.<br>
<br>
결과적으로 FC Layer 1x1 Convolution Layer로 바꿈으로 약간 큰 사이즈 이미지를 넣어고 Horizontal Flipping만 적용했기 때문에 빠르고 성능마저 좋은 효과를 얻었다고 한다.

<br>
<br>

## Single Scale Evaluation

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/0a7aed43-136f-45bf-932c-b053cef10ef4">
</p> 

<br>

고정 S에 대해서 Q = S, S = [S_min, S_max]에서의 임의 S에 대해선 Q = 0.5(S_min + S_max) 사용<br>

```
3 - Insight

1. A-LRN 모델은 LRN을 적용하지 않은 A 모델에 비해 성능 개선 X
    -> 더 깊은 모델인 B-E 모델에는 적용 X

2. 분류의 오차 감소는 Convolution Net의 깊이 증가와 함께 이루어지는 것 관찰

3. 데이터 증강 기법인 Scale Jittering (S=[256;512])을 적용했을 때, Single Scale을 사용했음에도
   고정된 크기 (S=256 or S=384)를 갖는 사진에 대해 훈련한 것보다 나은 성능 보여줌
   -> 이는 Scale Jittering을 통한 데이터 증강이 Multi-Scale Image 통계를 얻는데 도움됨을 확인
```


<br>
<br>

## Multi Scale Evaluation

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/b9e00ba5-cc2a-4a14-9fb5-ffb9aaee9da8">
</p> 

<br>

추가적으로 이 부분에서는 검증 데이터 또한 Scale Jittering 하는 것이 고정된 크기를 갖는 사진만 사요했을 때보다 개선된 성능을 보여주는 것을 알 수 있었다고 한다.


<br>
<br>

## Multi-Crop Evaluation

<br>

<p align="center">
  <img src="https://github.com/CKtrace/Research-Paper-Review/assets/97859215/659ef976-0ac1-44ec-8627-d1807d2415ad">
</p> 

<br>

해당 부분에서는 Multi-Crop Evaluation과 Dense Evaluation는 함께 사용 시 성능 개선이 됨을 알 수 있으며 이 둘은 상호보완적이라는 것을 알 수 있다.

<br>

이때, Dense Evaluation은 Croped Image마다 네트워크에 입력하는 것과는 달리 큰 이미지를 네트워크에 한번 입력한 다음 Sliding window를 적용하는 것과 같이 일정한 간격으로 결과를 도출할 수 있는 방법이다.

<br>
<br>

## Conclusion

<br>

본 연구를 통해 네트워크의 깊이는 분류의 정확도에 지대한 영향을 끼친다는 것을 다시 한 번 알 수 있었다고 한다.

<br>
<br>



