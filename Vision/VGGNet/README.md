# [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)

## Abstract

본 연구는 Large-Scale 이미지 인식에서 Convolutional Nerwork의 깊이가 주는 영향에 대해 설명한다.
<br>
3x3 Convolution Layer만 사용하여 Network의 깊이를 늘렸으며, 16-19개의 layer까지 늘리며 성능 개선에 성공하였다고 한다.


<br>

## Introduction

본 논문은 Convolution Network 구조에서의 깊이의 중요성 측면에 대해 서술한다.
<br>
연구진들은 Architecture의 여러 파라미터를 고치면서, 꾸준하게 더 많은 3x3 Convolution Filter를 사용하여 Network의 깊이를 늘려나갔다.

<br>

## Architecture


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


## Discussion

본 파트에서는 Single 7x7 Convolutional Layer 대신 three stack 3x3 Convolutional Layer를 사용하였을 때, 얻은 두 가지 이점에 대해 서술한다.

```
1. Single ReLU를 사용하는 대신 세 개의 ReLU를 사용하면 결정함수가 더 구체적이게 된다.
2. 파라미터 수를 줄일 수 있는데, If Channel == C,
    
    - Single 7x7 Convolutional Layer 사용 시 Param => 7**2(C**2) = 49C**2
    - Three Stack 3x3 Convolutional Layer 사용 시 Param => 3(3**2(C**2)) = 27C**2
    - 약 81% 더 줄일 수 있다.
    
```

C 모델에서의 1x1 Convolutional Filter는 타 연구에서 사용된 적이 있지만, 그들이 제안한 Network는 본인들이 제안한 모델들보다 얕았으며, Large-Scale ILSVRC 데이터셋을 사용하지도 않았다고 한다.
그리고 GoogLeNet과 비교하며, 자신들이 제안한 모델은 덜 복잡하고, 첫 번째 Layer, Feature Map에서의 Spatial Resolution이 더욱 과감하게 연산량을 감소시켰다고 한다.


<br>


## CLASSIFICATION FRAMEWORK

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

### Training Image Size

Training Scale을 S라 하고 해당 Training Image 224x224로 Crop시킨다.<br>
이때, S=224인 경우, 이미지의 전체 부분이 포함되겠지만, S>224인 경우 이미지의 작은 부분에 해당할 것이며, 작은 부분에 해당할 것이며, 작은 물체 또는 물체의 일부만 포함될 것이다.

<br>

S를 설정하기 위해 아래와 같은 두 개의 방법을 고안했다.
