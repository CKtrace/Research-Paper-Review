# [Convolutional Neural Networks for Sentence Classification (EMNLP 2014)](https://aclanthology.org/D14-1181.pdf)

<br>

## Abstract

<br>

본 논문에서는 pretrained 된 Word2Vec에 CNN을 사용한 sentence classification model을 제안한다.

```
Sentence Classification

1. 감정 분류 (Sentiment Analysis)
2. 주제 분류 (Subjectivity Analysis)
3. 질문 분류 (Question Analysis)
```

간단한 CNN과 약간의 hyperparameter tuning, static vector로 여러 Benchmark에서 훌륭한 결과를 도출했다.

추가적으로 간단한 모델 구조의 변형으로 task-specific과 static vectors의 사용 가능한 방법을 제안한다.

Sentiment Analysis & Question Classification을 포함한 7가지 tasks 중 4개의 부분에서 state-of-art를 거뒀다.

<br>

## Intoduction

<br>

CNN은 NLP에서 효과적인 모델이라는 것을 보여줌과 동시에 semantic parsing, sentence modeling, 다른 전통적인 NLP tasks에서 좋은 성능을 보여주었다.

본 연구에서는 하나의 Convolutional layer 위에 Word2Vec가 있는 간단한 CNN 모델을 선보인다.

적은 hyperparameter tuning으로 여러 Benchmark에서 좋은 성능을 보였다.

<br>

## Model 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/d3687c73-435a-4779-b089-72b11416058d" width='700' height='300'>

</p> 

<br>

```
n x k representation of sentence
n : count of concat words
k : dimensional word vector

Conv Filter -> w
h : window of h words (window height)
k : window width (dimensional word vector) 

c_i = f(w * x_i:i+h-1 + b)

c = [c_1, c_2, ... , c_n-h+1]

f -> tanh function

Max-over-time pooling
(Reason why they use : To capture the most important feature)
c^ = max{c}
```

n x k의 matrix는 두 채널로 존재하는데, 하나는 학습 동안 단어 벡터를 고정한 것이고, 하나는 역전파를 통해 파인 튜닝 될 수 있도록 한 것이다. 

<br>

### Regularization

<br>

끝에서 두 번째 레이어 Dropout을 사용, 이를 통해 특정 feature에 편향되지 않게 한다.

또한 아래와 같은 수식의 변화가 존재한다. 

```
y = w x z + b -> y = w(z o r) + b

r : 'Masking' vector로 Bernoulli random variables
```



<br>

## Datasets and Experimental Setup

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a7d628fc-460b-46b1-b7c7-7e7a5420ed6c" width='400' height='400'>

</p> 

<br>

Datasets은 MR, SST-1, SST-2, Subj, TREC, CR 데이터셋을 사용했다.

```
MR : 리뷰 당 한 문장인 영화 리뷰 데이터. 리뷰 긍정/부정.

SST-1 : 스탠포드 감정 트리뱅크(Treebank) - MR의 확장 버전 느낌. train/dev/test가 나뉘어 있고, 감성이 좀 더 세밀하게 라벨링 되어 있음(아주 좋음, 좋음, 중립, 나쁨, 아주 나쁨)

SST-2 : 위와 똑같지만 중립인 리뷰가 제거 -> 이진 분류

Subj : 문장이 주관적/객관적 분류하는 데이터셋

TREC : TREX 질문 데이터셋 - 질문을 6개의 종류(about person, numeric, location..)로 분류

CR : 다양한 제품에 대한 소비자 리뷰. 긍정인지 부정인지 맞추는 태스크

MPQA : MPQA데이터셋에서 하위 태스크인 Opinion polarity detection
```

<br>

### Hyperparameters and Training

<br>

```
1. RELU

2. Windows (h) of 3, 4, 5 with 100 feature maps each

3. Mini-Batch size = 50

4. L2 constraint = 3

5. Dropout Probability = 0.5

-> 이 값들(1~5)은 SST-2 val data에서 그리드 서치를 통해 찾았다고 한다.
```

<br>

### Pre-trained Word vectors

<br>

1000억 개가 넘는 단어들로 구성된 Google News로 학습 Word2Vec의 vector를 사용했다.

해당 vector는 300개의 차원을 갖고 CBOW를 이용해 학습한다.

사전 학습된 단어 집합에 없는 단어는 랜덤하게 초기화하였다고 한다.

<br>

### Model Variations

<br>

#### CNN-rand

모든 단어들이 랜덤하게 초기화되어있고 학습 과정에서 수정되는 방식을 띄는 baseline model이다.

<br>

#### CNN-static

Word2Vec로부터 pre-trained된 단어 벡터 사용하였으며, 사전 학습된 단어 집합에 없는 단어는 랜덤하게 초기화 하였다.

학습하는 동안 모든 단어는 변동 없고, 모델의 파라미터만 학습하였다

<br>

#### CNN-non-static 

CNN-static과 동일하지만 각 task에 맞춰 사전 학습된 단어 벡터를 사용된다.

<br>

#### CNN-multichannel

두 개의 word vectors 세트를 이용한다.

각 벡터 집합은 채널로 취급하고, 두 채널은 word2vec로 초기화된다.

하지만 역전파는 한 채널에만 적용하여 한 채널은 파인 튜닝이 됐지만 한 채널은 static을 유지한다.


<br>

위와 같은 변화 이외의 임의의 요소 효과를 제거하기 위해 CV-fold 할당 및 단어 집합에 없는 단어 초기, CNN Parameter 초기화와 같은 임의성을 갖는 요소들을 균일하게 유지한다.


<br>


## Results and Discussion

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/986d4e69-5f58-4d6d-9ad9-f519b8c6e386" width='800' height='600'>

</p> 

<br>

본 연구에서 제안하는 모델의 baseline인 CNN-rand는 성능이 좋지 않았다.

하지만 기대했던 word2vec를 사용한 모델의 성능은 기대 이상의 성능을 보여주었다고 한다.

간단한 CNN에 Word2Vec의 vector를 사용한 CNN-static의 성능도 DCNN, RNTN에 버금가는 성능을 보여주었다.

이러한 결과는 pretrained된 vector를 사용하는 것이 성능에 좋은 영향을 미치며, 'universal' feature extractors이고, 이는 여러 데이터셋에서 활용할 수 있다는 것을 알 수 있다.

마지막으로 각 task에 맞게 파인 튜닝해서 사용하는 것이 대체로 더 좋은 성능을 보이는 것을 알 수 있다.

<br>

### Multichannel vs Single Channel Models

<br>

다채널 구조가 오버피팅을 방지해서 단일 채널 구조보다 특히 작은 데이터셋에서 더 좋은 성능을 낼 것이라 예상했지만, 결과는 두 구조가 비슷했다고 한다.

<br>

## Conclusion

<br>

Word2Vec로 사전 학습된 단어 벡터를 사용한 간단한 CNN 구조를 제안한 본 논문의 모델은 약간의 하이파라미터 튜닝, 그리고 1개의 layer를 가진 CNN으로 뛰어난 성능을 보여주었다.

그 뿐만 아니라, 비지도 학습에서 사전 학습된 단어 벡터는 NLP에서 중요한 역할을 한다는 것을 알 수 있다.

<br>
