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

To be Updated...
