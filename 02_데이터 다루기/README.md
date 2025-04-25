# 정리


1. **훈련 세트와 테스트 세트로 나누어 사용하기**

   넘파이로 데이터를 정리하고, 사이킷런으로 훈련, 테스트 세트를 나누는 실습


2. **정교한 결과 도출을 위한 데이터 전처리**

   **정규화(표준 점수)** 란 통계 개념을 활용해서 모델이 더 정확하게 학습할 수 있도록 전처리 과정 실습

   <br/>

<details>
<summary> 훈련 세트와 테스트 세트로 나누어 사용하기</summary>

## **Numpy** 란?

> Numpy를 통해 데이터를 섞고 훈련세트와 테스트 세트를 셋팅

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

> 훈련 세트 (Training Set) 는 배우는 데 쓰고, 테스트 세트 (Test Set)는 성능을 확인하는 데 사용

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

</details>

<details>
<summary> 정교한 결과 도출을 위한 데이터 전처리 </summary>

## Numpy 활용하여 전처리

> 모델이 학습할 수 있는 형태로 변환한 것이 바로 데이터 전처리의 시작

```py
fish_data = np.column_stack((length, weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
```

- `column_stack` : 두 개 이상의 배열을 세로로 붙여서 **열(column)** 기준으로 합치는 메서드

    - `[[길이, 무게]]` -> `[[length1, weight1], [length2, weight2], ...]` 형태의 2차원 배열 생성
    
    <br/>

- `concatenate` : 두 배열을 하나로 이어 붙여주는 메서드

## 사이킷런(sklearn)으로 훈련 세트 / 테스트 세트 분리

```py
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
```

- `train_test_split()` : 데이터를 **훈련용** 과 **테스트용** 로 자동으로 랜덤 분할해주는 메서드

   <br/>

- 파라미터 설명

   - `fish_data` : 전체 입력 데이터 [길이, 무게]

   - `fish_target` : 정답 데이터 (도미 : `1` , 빙어 : `0`)

   - `train_input` , `train_target` : 훈련용 데이터 와 정답 데이터

   - `test_input` , `test_target` : 테스트용 데이터와 정답

   - `stratify=fish_target`: **도미와 빙어의 비율을 훈련세트 와 테스트세트 모두에 동일하게 유지**

       ```py
       fish_target = np.concatenate((np.ones(35), np.zeros(14)))
       # 위 `fish_target` 을 `stratify` 의 기준으로 하면
       ```

       ```py
       train_test_split(..., stratify=fish_target, test_size=0.25)
       ```

       - `test_size` : 전체 데이터의 25%를 테스트셋으로 뽑는다면

       - 훈련세트 : 약 75% (도미 26마리, 빙어 10마리)

       - 테스트세트 : 약 25% (도미 9마리, 빙어 4마리)
       
       <br/>

   - `random_state=42` :  데이터 분할을 재현 가능하게 고정하는 역할, 숫자는 아무거나 사용 가능 -> 때 사용하는 **랜덤 시드(seed)** 를 고정

      - 랜덤 시드가 왜 필요?

         > 모델의 학습과정이나 초기화 과정에서 발생하는 난수들 때문에 모델의 결과가 불안정하게 변할 수 있기 때문입니다. (kjws0712 블로그 참고)

         
         > 예를 들어, 동일한 데이터와 모델 아키텍처를 사용하더라도 초기 가중치나 데이터 배치 등에서 발생하는 난수에 따라 모델의 학습 결과가 달라질 수 있습니다. 이러한 불안정성은 모델의 성능을 평가하거나 비교하기 어렵게 만들 수 있습니다. (kjws0712 블로그 참고)

         결과가 고정, 실험을 반복해도 일관된 결과를 얻기 가능

## KNN 훈련 및 예측

이번 실습은 **K-최근접 이웃(KNN) 알고리즘** 에서 예측 대상이 되는 데이터 포인트와 가장 가까운 이웃(= 훈련 데이터 중 가장 비슷한 데이터들)을 시각화 과정을 진행

- `kneighbors()` : 가까운 이웃들의 **거리** 와 **인덱스** 를 구함

- `plt.scatter()` : 전체 데이터, 예측 대상, 가까운 이웃을 그래프로 시각화

```py
distance, indexes = kn.kneighbors([[25, 150]]) # 1

plt.scatter(train_input[:,0], train_input[:,1]) # 2
plt.scatter(25, 150, marker= '^') # 3
plt.scatter(train_input[indexes, 0],train_input[indexes, 1], marker= "D") # 4
```

1. `kn.kneighbors([[25, 150]])` : 입력 데이터 기준으로 가장 가까운 `k`개의 이웃 데이터를 찾아서 반환

   - `KNeighborsClassifier` 모델인 `kn` 에서 `[25, 150]` 새로운 데이터를 넣음

   - 가장 가까운 훈련 데이터 5개 (기본값 `5`)

   <br/>

2. `plt.scatter(train_input[:,0], train_input[:,1])` : 전체 훈련 데이터를 산점도로 그림

   - `train_input[:, 0]` : 길이 `x`축

   - `train_input[:, 1]` :  무게 `y`축

   <br/>

3. `plt.scatter(25, 150, marker= '^')`

   - 예측하려는 새로운 데이터 `[25, 150]`을 `^`로 표시 (이 물고기는 도미니? 빙어니?)

   <br/>

4. `plt.scatter(train_input[indexes, 0],train_input[indexes, 1], marker= "D")`

   - KNN이 이 데이터들과 얼마나 가까운지를 기준으로 `[25, 150]` 의 정답(도미 인지 빙어)을 예측

## 데이터 전처리(표준화)

> 표준화(Standardization) 란 규격에 맞춰 무언가를 균일하게 만드는 것

1. 기준 변경

   ```py
   plt.xlim((0, 1000))
   ```

   x축의 범위를 0부터 1000까지 설정
   
   <br/>

2. 표준 점수로 바꾸기 (정규화/표준화)

   ```py
   mean = np.mean(train_input, axis=0)
   std = np.std(train_input, axis=0)
   ```

   - `mean` :  열(길이, 무게)의 평균값 -> `[평균 길이, 평균 무게]`

   - `std` : 각 열의 표준편차 -> 얼마나 데이터가 퍼져 있는지

   ```py
   train_scaled = (train_input - mean) / std
   ```

   - 각 훈련 데이터를 평균으로 빼고 표준편차로 나누면
평균이 `0` , 표준편차가 `1` 인 스케일로 맞춰짐 

   <br/>

3. **수상한 도미 다시 표시하기**

   ```py
   new = ([25, 150] - mean) / std # 1

   plt.scatter(train_scaled[:,0], train_scaled[:,1]) # 2
   plt.scatter(new[0], new[1], marker='^')
   plt.xlabel('length')
   plt.ylabel('weight')
   plt.show()
   ```

   1. `[25, 150]` 이라는 새로운 데이터도 동일하게 표준화

   2. 표준화된 훈련 데이터와 수상한 도미의 위치를 산점도로 보여줌

   <br/>

4. **전처리 데이터에서 모델 훈련**

   ```py
   kn.fit(train_scaled, train_target) # 1

   test_scaled = (test_input - mean / std) # 2
   kn.score(test_scaled, test_target) # 3

   print(kn.predict([new]))

   # 4 나머진 동일
   distance, indexes = kn.kneighbors([new])

   plt.scatter(train_scaled[:,0], train_scaled[:,1])
   plt.scatter(new[0], new[1], marker='^')
   plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
   plt.xlabel('length')
   plt.ylabel('weight')
   plt.show()
   ```

   1. **전처리된 데이터로 모델 훈련**
   
      `kn.fit(train_scaled, train_target)` : 기존의 kn 모델을 표준화된 데이터로 다시 학습

      <br/>

   2. **테스트 데이터도 같은 방식으로 표준화**

      `test_scaled = (test_input - mean) / std`

      <br/>

   3. **테스트 정확도 확인**

      `kn.score(test_scaled, test_target)`

## 표준화를 왜 하는 걸까?

길이/무게처럼 크기 단위가 다른 특성들은 전처리(표준화)를 안 하면 가까움을 비교할 때 왜곡되기 때문

</details>


## 참고

- [블로그 참고 - 불곰(Numpy란?)](https://brownbears.tistory.com/480)

- [블로그 참고 - 개발자로 취직하기:티스토리(Python.NumPy ndarray 이해하기)](https://coding-grandpa.tistory.com/24)

- [블로그 참고 - kjws0712(random_state를 머신러닝에서 사용하는 이유)](https://kjws0712.tistory.com/118)

- [블로그 참고 - iphoong(분류(Classification) 알고리즘)](https://iphoong.tistory.com/6)

- [블로그 참고 - sungwookoo(정규화 vs 표준화)](https://sungwookoo.tistory.com/35)
