# [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Neural Computation 1997)

<br>

오래되고 중요한 논문인만큼 논문의 흐름보다는 이해한 내용을 바탕으로 작성하도록 한다.

<br>

## Vanilla RNN Structure

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/a61d00ce-1f6d-4734-a04e-8c07c6d5167f" width='700' height='250'>

</p> 

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/2499df7d-3f1e-41a4-84e8-44ce4d567ffa" width='400' height='300'>

</p> 

<br>

## Existing RNN Shortcoming

<br>

Long-time Step일수록, 학습이 불가해지는 단점이 존재한다.

Vanishing Gradient & Exploding Gradient -> BPTT 활용으로 인해 발생

<br>

## Long Short-Term Memory

<br>

하나의 Unit에서 계산된 정보(Short-Term)을 길게(Long) 전파하는 Memory 구조 제안

<br>

### 3 Contribution

<br>

1. 장기 기억 Cell (CEC : Constant Error Carousel) 도입해 Vanishing Gradient
2. Gate 구조 도입
3. Training Method 개선 (본 논문에서는 BPTT+RTRL)

<br>

### 본 논문에서 제안한 LSTM 구조

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/29e764d9-ebe1-4969-9202-88c6c153d71c" width='400' height='300'>

</p> 

<br>

CEC가 LSTM의 핵심이라고 볼 수 있다.

왜냐하면, CEC는 기존 RNN의 Gradient Problem을 해결해냈기 때문이다.

하지만, CEC는 identity mapping이라 Exploding Gradient Problem을 겪기에 추후 연구에서 아래와 같은 forget gate를 추가한 구조가 제안된다.

<br>

<p align="center">

  <img src="https://github.com/user-attachments/assets/411880bc-64b9-4009-bfc4-df44e2e535cd" width='400' height='300'>

</p> 

<br>

현재 사용되는 LSTM은 squash 함수가 tanh으로 대체되었다.

