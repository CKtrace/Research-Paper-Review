# [Enriching Word Vectors with Subword Information(ACL 2017)](https://arxiv.org/pdf/1607.01759)

<br>

## Abstract

<br>

Large unlabeled corpora로 학습되어 만들어진 Continuous word representations는 Language Processing Task에서 유용하다.

Continuous word representations를 활용해 학습한 유명한 모델들은 단어의 형태학적 특징을 무시한다.

이것은 특히 large vocabularies와 많은 rare words가 있는 language에 limitation이 된다.

해당 논문에서 제안하는 모델은 Skip-gram 모델을 Base로 삼는다.

이들이 제안한 방법은 large corpora를 빠르고 training data에 없던 word의 word representations까지 계산할 수 있다고 한다.

제안한 word representations 방법은 단어 유사도와 분석 task에서 9개의 language로 평가하였다고 한다.

그들이 제안한 방법은 최근에 제안된 morphological word representation과 비교해 state-of-art performance를 보인다고 한다.

<br>

## Introduction

<br>

최신 연구들을 보았을 때, vocabulary에 존재하는 단어들을 parameter sharing 없이 distinct vector로 표현하였다.

이러한 점은 중요한 한계점을 가져오는데, 그것은 바로 형태학적으로 변환이 많은 언어들인 터키어, 핀란드어와 같은 언어들에 대해 취약하다는 것이다.

형태적 변환이 많은 단어들을 지니고 있는 언어들의 말뭉치에는 그것들이 많이 등장할테고 그러한 점은 좋은 word representation을 학습해내는데 제약을 갖는다.

해당 논문에서는 n-grams으로 표현된 representation을 학습하는 방법과 n-gram vector의 합으로 단어를 표현하는 방법을 제안한다.

제안하는 방법은 Skip-gram model의 연장인 격이다.

<br>

## Related Work

<br>

### Morphological word representations

<br>

Morphological information을 word representations에 포함시키려는 많은 시도들이 있었다.

여러 시도들의 접근 방법은 다르지만 단어의 morphological decomposition에 의존하는 것은 공통점이며, 해당 연구는 그렇지 않다고 한다.

또한 character n-grams를 해당 연구에서 사용하는데, 이전 연구 중에도 이를 사용한 방법들이 제안되었지만 그 연구들은 paraphrase pairs에 기반을 두고 학습을 했지만, 본 연구에서 제안하는 모델은 어떤 text corpus로 학습해도 된다고 한다.

<br>

## Model

<br>

해당 섹션에서는 일반적인 모델로 word vectors를 학습하고, 본 연구에서 제안하는 Subword model과 어떻게 dictionary의 character n-gram을 활용했는지에 설명한다.

<br>

### General Model

<br>

해당 섹션에서는 본 연구에서 제안하는 모델의 기반이 되는 Skip-gram에 대해 설명하고 있다.

과거의 연구에 따르면, context의 각 단어의 벡터 표현은 예측을 위한 학습이 잘 될 수 있도록 하는 것을 밝혔다.

Skip-gram은 한 단어의 주변 단어들을 예측하는 모델이다.

Skip-gram은 Negative Sampling 기법을 사용하는데, 이는 주변 단어가 아닌 멀리 떨어져 있는 단어를 일부러 넣는 것이고, 이로 인해 Skip-gram은 주변 단어인지 아닌지에 대해 binary하게 classification 하게 된다.


<br>

### Subword

<br>

Skip-gram은 distinct vector representation을 사용하기에, 단어의 내부 구조(morphology)를 무시하게 된다. 

해당 연구에서는 단어를 bag of character n-gram으로 표현하는데, 다른 character sequence로부터 접미사와 접두사를 구별하기 위해 < and > 라는 symbol들을 붙인다.

또한 n-grams의 set에 단어 자체도 넣어준다.

```
만약, where라는 단어의 character n-grams의 set을 구한다고 해보자. n=3

해당 set는 아래와 같다.

<wh, whe, her, ere, re, where>

Special Sequence = where
```

n-grams의 size가 G라고 했을 때, {1, ... , G}까지의 n-grams들이 나올텐데 이를 Summation한 벡터를 사용하게 되면, rare한 단어들도 잘 학습할 수 있게 된다.

제안한 모델에 요구되는 메모리 때문에, hashing function을 이용해 n-grams을 1부터 K의 정수의 크기로 mapping 하였다. 

이때, 그들은 Fowler-Noll-Vo hashing function을 사용하였으며, K = 2.10^6으로 설정하였다고 한다.


<br>

## Experiment Setup

<br>

제안한 모델을 비교하기 위한 Baseline model로 Word2Vec 패키지의 CBOW와 Skipgram을 사용한다고 한다.

English data를 사용하였는데, 해당 character n-gram을 사용한 제안된 모델은 Skipgram보다 학습하는데 1.5배 느렸다고 한다.

데이터셋은 Wikipedia data를 사용했으며, 해당 데이터에서 9개의 언어 데이터를 다운 받았으며 해당 실험에서 논문에서 제안한 모델은 4개의 언어 데이터를 이용해 학습을 진행했다고 한다.

<br>

## Results

<br>

Training datasets에는 없는 단어이지만, test set에는 있는 단어를 OOV(Out of Vocabulary) Word라고 하는데,character n-gram을 사용하는 FastText와 달리 cbow와 skipgram으로는 해당 단어를 사용하지 못한다.

OOV word를 null vector로 하고 진행하는데, 같은 선상에서의 비교를 위해 OOV word를 null vector로 처리하여 학습한 FastText를 sisg-, 그렇지 않고 제안한 FastText 그대로 사용하는 것은 sisg로 표기하였다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/88934a75-bbd6-47a5-bc2b-f493677c209b" width='300' height='500'>

</p> 

<br>

해당 결과를 통해 OOV word까지 처리할 수 있는 character n-gram을 적용한 sisg가 그렇지 않은 sisg-보다 결과가 좋다.

이는 Subword Information이 학습에 미치는 영향에 대해 알 수 있게 한다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/040713da-4ab9-4b86-8878-69de9bc78ddc" width='300' height='300'>

</p> 

<br>

해당 결과는 English Rare Words dataset을 이용한 것인데, Frequency가 낮은 단어들을 학습하니 Skip-gram과 CBOW의 성능이 FastText에 비해 떨어지는 것을 확인 할 수 있다.

이는 Character Level에서의 유사도는 Good word vector를 학습할 수 있도록 도움을 준다는 것을 알 수 있다.

이외에 여러 평가들을 진행했는데, 평가들을 통해 얻은 insight는 아래와 같다.

```
1. CBOW와 다르게 FastText는 데이터가 많다고 해서 결과가 항상 향상되는 것은 아니다
2. FastText는 적은 dataset으로도 Very good word vectors를 구할 수 있게 한다.
3. 접미사와 접두사에 강해 학습 데이터에 없는 단어들에도 Robust한 모습을 보인다.
```

<br>

## Conclusion

<br>

Character n-grams를 적용한 Skip-gram인 FastText는 morphological problem에 Robust에 강하다.

<br>