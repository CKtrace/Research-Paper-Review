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

