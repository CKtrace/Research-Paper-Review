# Finding Structure in Time (CogSCI 1990)

<br>

## Abstract

<br>

Connectionist Model에서 시간성을 부여하는 것은 매우 중요하다.

기존의 연구 중, 동적 메모리로 네트워크를 제공하기 위해 recurrent link를 사용하여 이전의 상태를 현재에 반영하도록 한 연구가 있었다.

해당 방법은 단어로부터 Syntatic/Semantic feature를 찾는데도 도움을 준다고 한다.


<br>

## Introduction

<br>

시간은 Language와 같은 많은 행동의 temporal sequence와 뗄 수 없는 관계이다.

Natural Language에서는 parsing을 통해 temporal sequence를 지켜내려고 했지만 그것은 자명한 해결책 아니다.

위 문제에 대한 괜찮은 접근 방법은 시간을 명시적으로 표현하는 것보다 암묵적으로 표현하는 것이다.

본 연구에서도 해당 방법으로 접근하였는데 명시적으로 표현한다는 것은 직접 input 값을 추가로 주어 시간을 표현하는 방법이고 암묵적으로 표현하는 것은 추가적인 input 없이 시간성을 부여하겠다는 것이다.

해당 논문에서는 위에서 설명한 접근 방식으로 새로운 architecture에 Natural Language Data에서 syntatic/semantic 카테고리를 찾는데 적용해보고자 한다.

<br>

## The Problem With Time

<br>

첫 번째 접근 방법은 시간을 명시적으로 표현하는 것이다.

즉, 시간을 표현하는 벡터를 기존의 데이터와 별개로 추가적으로 input으로 넣겠다는 것이다.

해당 방법은 시스템이 언제 시간 정보가 담긴 데이터를 대응해야 하는지 불분명한 것과 같은 여러 가지 문제가 존재한다.

또한 입력 길이 역시 longest possible pattern에 맞춰 설정되기에 시간 데이터의 길이 역시 제한이 되어 있다.

이러한 문제점들은 특히 Language와 같은 도메인에서 특히 더 문제가 된다.

마지막으로 아래와 같은 두 벡터가 존재한다고 가정한다.

```
[0 1 1 1 0 0 0 0 0]
[0 0 1 1 1 0 0 0 0]
```

두 벡터는 같은 패턴이지만, 공간적 또는 시간적 정보에서는(즉, 기하학적 관점) 두 벡터는 유사하지 않다고 본다.

본 논문에서 제안하는 방법은 간단한 architecture로 시간 표현을 풍부하게 하고, 이러한 문제점들 또한 갖고 있지 않는다고 한다.

<br>

## Networks with Memory

<br>

해당 논문에서 제안하는 방법은 현재 process에 시간적 표현의 영향을 주는 것이다.

즉, processing system에 대응되는 temporal sequence를 준다는 것이다.

이 방법에는 여러 가지 접근 방법에 대한 연구들이 이어졌는데, 그 중 recurrent connection을 포함하는 방법이 가장 유망한 방법이라고 한다.

Recurrent Connection은 Network의 hidden network이 이전의 output을 통해 후속 output이 이전의 output에 영향을 받을 수 있게 한다.

이 접근 방법을 다음과 같이 수정하여 적용할 수 있다.

```
Network의 input level에 'Context Units'을 추가하고, 해당 유닛은 'hidden' 유닛으로 Network의 내부에 존재한다. 
```

해당 architecture는 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/0a3f3a1a-7e73-402c-8eb4-a6de6f80b149" width='500' height='700'>

</p> 

<br>

Context Unit은 학습이 진행되면서 time representation을 개선해 나간다.

학습되는 전반적인 architecture는 아래와 같다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/77f17622-5daa-48f3-b911-ba5d9dd6bacc" width='500' height='700'>

</p> 

<br>

## Conclusion

<br>

해당 architerture의 가장 큰 문제점은 장기 의존성 문제이다. (The Problem of Long-Term Dependencies)

이를 추후에 읽게 될 LSTM에서 해결한다.

<br>