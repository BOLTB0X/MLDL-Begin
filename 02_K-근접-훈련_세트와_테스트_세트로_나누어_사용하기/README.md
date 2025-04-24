# 정리

Numpy를 통해 데이터를 섞고 훈련세트와 테스트 세트를 셋팅

훈련 세트 (Training Set) 는 배우는 데 쓰고, 테스트 세트 (Test Set)는 성능을 확인하는 데 사용

## **Numpy** 란?

- 과학 계산용 파이썬 라이브러리

   <br/>


- 수학, 통계, 머신러닝, 딥러닝에서 쓰이는 **벡터/행렬** 계산에 주로 쓰임

   - 다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록 지원

      <br/>


- 핵심은 `ndarray` 라는 배열 자료형

   - **NumPy** : N차원 배열 객체

   - `np.array` 함수: 기존의 리스트 형태의 데이터와 데이터 타입을 입력 받아 `ndarray` 형태로 변환해주는 함수

      <br/>


## List와의 차이

```py
import numpy as np

# 파이썬 List
a = [1, 2, 3]
b = [4, 5, 6]

print(a + b)  # List 덧셈은 이어붙이기: [1,2,3,4,5,6]

# Numpy 배열
na = np.array([1, 2, 3])
nb = np.array([4, 5, 6])

print(na + nb)  # Numpy 배열 덧셈은 요소별 더하기: [5 7 9]
```

항목 | 파이썬 리스트 (`list`) | 넘파이 배열 (`np.array`)
| ---| ------------------ | -------------------
자료 구조 | 다양한 자료형 가능 (`int` , `str` 혼합 가능) | 하나의 자료형만 가능 (`int` , `float` 등 통일)
속도 | 느림 (루프 많이 돎) | 빠름 (C로 구현된 내부 연산)
메모리 | 비효율적 | 효율적 (고정 크기 타입)
연산 | 직접 루프 돌려야 함 | 벡터/행렬 연산 가능
사용 목적 | 일반적인 데이터 저장 | 수치 계산, 과학 계산, 머신러닝/딥러닝

   <br/>

Numpy 는 타입, 차원, 연산 조건에 민감


1. 차원 불일치
 
   ```py
   a = np.array([1,2,3])
   b = np.array([[1,2],[3,4]])
   a + b  # ValueError: operands could not be broadcast
   ```
      <br/>


2. 스칼라 vs 벡터 연산

   ```py
   a = np.array([1,2,3])
   print(a + 1)  # [2 3 4] 이건 가능
   ```

## 훈련 세트 (Training Set)

- 말 그대로 모델을 훈련(train) 시키는 데 사용하는 데이터

   <br/>


- 머신러닝 모델이 여기 있는 **입력(X)과 정답(y)** 을 보고 패턴을 학습

   <br/>


```py
index = np.arange(49)
np.random.shuffle(index)

print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
```

## 테스트 세트 (Test Set)

- 훈련이 끝난 모델이 얼마나 잘 배웠는지 평가할 때 사용하는 데이터

   <br/>


- 테스트 세트는 모델이 한 번도 본 적 없는 데이터여야 한다고 함(현재 실습은 제외)

##  훈련/테스트 세트로 나눈 후 input과 target 구분하는 이유?

- input

   모델이 보는 데이터 (입력, X) : `[[25.4, 242.0], [26.3, 290.0], ...]`

   <br/>

- target 

   정답(label), 모델이 맞혀야 하는 것 : `[1, 1, ..., 0, 0]` (도미: `1` , 빙어: `0` )

   <br/>


- 코드 흐름

   1. 목표 : 생선의 길이와 무게(입력) -> 도미인지 빙어인지(정답)를 맞히는 것

      ```py
      fish_data = [[l, w] for l, w in zip(length, weight)]
      fish_target = [1] * 35 + [0] * 14 # 도미 (1) , 방어 (0)
      ```
      
      <br/>

   2. 훈련/테스트 세트로 나눈 후 **input** 과 **target** 구분

       ```py
       train_input  = fish_data[:35]   # 도미 35마리
       train_target = fish_target[:35] # 도미 정답들 (1)

      test_input  = fish_data[35:]    # 빙어 14마리
      test_target = fish_target[35:]  # 빙어 정답들 (0)
      ```

      이걸 사용하여

      ```py
      kn.fit(train_input, train_target)     # 훈련할 때는 train 데이터로 학습
      kn.score(test_input, test_target)     # 테스트할 때는 test 데이터로 평가
      ```

      **input** 과 **target(정답)** 은 모델의 학습과 평가에서 각각 다른 역할을 하기 때문에 꼭 구분해서 관리


## 참고

- [블로그 참고 - 불곰(Numpy란?)](https://brownbears.tistory.com/480)

- [블로그 참고 - 개발자로 취직하기:티스토리(Python.NumPy ndarray 이해하기)](https://coding-grandpa.tistory.com/24)