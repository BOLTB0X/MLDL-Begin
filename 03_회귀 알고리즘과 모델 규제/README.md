# 정리

<details>
<summary> Intro 회귀 </summary>

## 회귀(Regression)

> **숫자(연속된 값)** 를 예측하는 문제를 통틀어 **회귀 문제** 라 함

- 키에 따른 몸무게 예측 -> 몸무게는 숫자이므로 회귀

- 공부 시간에 따른 시험 점수 예측 -> 점수도 숫자이므로 회귀

- 자동차 주행 거리 예측 -> 마찬가지로 숫자니까 회귀

## 왜 ‘회귀’라는 말을 쓰게 되었나

> *19세기 통계학자 프랜시스 골턴(Francis Galton) 이 부모의 키와 자녀의 키 사이의 관계를 연구하면서 처음 사용*

- 처음

   - 부모의 키가 크면 자녀도 크지만, 완전히 똑같이 크지는 않음

   - 평균적으로 부모보다 자녀의 키가 평균 쪽으로 돌아가는(회귀하는) 현상이 나타났음
   
   - **regression toward the mean (평균으로의 회귀)** 라고 부름
   
   <br/>

- 현재

   - 어떤 숫자(연속적인 값)를 예측할 때 사용하는 알고리즘을 통틀어 부르는 말

      - **선형 회귀 (Linear Regression)** : 직선을 그려서 숫자 예측

      - **K-최근접 회귀 (K-Nearest Neighbors Regression)** : 주변 데이터 평균으로 예측

      - **결정 트리 회귀 (Decision Tree Regression)** : 트리 구조로 숫자 예측

   <br/>

## 분류(Classification) vs 회귀(Regression)

**분류(Classification)** : 정답이 **카테고리** 인 경우 (ex. 도미/방어)

**회귀(Regression)** : 정답이 **숫자** 인 경우 (ex. 길이, 무게, 가격 등)

## K-근접 이웃 (KNN) 과 K-근접 회귀 (KNN Regression)

1. **K-근접 이웃 (K-Nearest Neighbors Classification)**

   > 분류 문제를 해결하는 데 사용됨 ex: 이 물고기는 도미일까? 빙어일까?

   - 동작 흐름

       1. 예측하고 싶은 새로운 데이터가 있을 경우

       2. 훈련 데이터 중에서 **가장 가까운 K개의 데이터** 를 찾음

       3. 그 K개의 데이터 중 **가장 많이 등장한 클래스(label)** 를 정답으로 선택함

       <br/>

    - 예시: `[25, 150]` 이라는 생선이 있다면
    
       1. 주변 생선 5개 중 도미가 4개, 빙어가 1개
       
       2. 결과는 **도미(1)** 로 예측

       <br/>

2. **K-근접 회귀 (K-Nearest Neighbors Regression)**

   > 수치 예측 문제를 해결할 때 사용됨 ex: 도미의 길이는?

   - 동작 흐름

      1. 예측하고 싶은 **새로운 데이터** 가 있음

      2. 훈련 데이터 중에서 **가장 가까운 K개의 데이터** 를 찾음

      3. 그 K개의 데이터의 **타겟 값(숫자)** 을 평균울 내어 **예측값**으로 사용함

      <br/>

    - 예시: `[25, 150]` 이라는 생선이 있다면

       1. 주변 유사한 5개 도미 `[25.4, 26.3, 26.5, 29.0, 29.0]`

       2. 예측값 0.6 , **"도미일 확률이 60%"** 

       <br/>

</details>


<details>
<summary> 회귀문제를 이해하고 k 최근접 이웃 알고리즘으로 풀어보기 </summary>

> 분류는 다수결, 회귀는 평균 이라는 것

| 항목 | K-최근접 분류 (`KNeighborsClassifier`) | K-최근접 회귀 (`KNeighborsRegressor`) |
| ------ | ------ | ----- |
| 목적 | 분류 (클래스 예측) | 수치값 예측 (연속적인 값) |
이웃들의 label | 다수결 투표로 클래스 결정 | 평균(또는 가중평균)으로 값 결정 |
| 예시 | 고양이냐 강아지냐 | 무게, 가격, 키 등 수치 예측 |
| 출력 | 클래스 이름 (ex. '도미', '빙어') | 연속적인 숫자 (ex. 250g, 500g) |


## 데이터 분리

```py
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
```

- 데이터를 훈련세트(train) 과 테스트세트(test)

- `random_state=42` 는 랜덤 동작의 출발점을 고정
  
   - 항상 같은 무작위 결과를 얻음

## 입력 데이터 형태 변경

```py
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

- `KNeighborsRegressor` 은 2차원 배열을 원함

- (샘플 수, 특성 수) 형태로 변환

## K-최근접 회귀 모델 만들고 훈련

```py
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
```

- 기본 `n_neighbors=5` 사용 (주변 5개 이웃 평균)

- 훈련 데이터로 모델 학습

## 평균 절댓값 오차(MAE) 계산

```py
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
# 19.157142857142862
```

- 오차가 얼만큼 나는지를 직관적으로 알 수 있게 해주는 지표

- 예측값과 실제값의 차이를 절댓값으로 평균낸 것

- 작을수록 좋은 모델

## 과대적합 vs 과소적합

```py
knr.score(train_input, train_target)
knr.score(test_input, test_target)
```

- 훈련셋 점수와 테스트셋 점수 비교

   - 훈련점수 높고 테스트점수 낮으면 -> **과대적합(overfitting)**

   - 둘 다 낮으면 -> **과소적합(underfitting)**

## K(이웃 개수) 줄이기

```py
knr.n_neighbors = 3
knr.fit(train_input, train_target)

print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```

- 이웃 수를 `5 -> 3`으로 줄임

   - **K를 줄이면 더 가까운 이웃만 보기 때문에 훈련 점수는 올라가고 테스트 점수는 달라질 수 있음**

   - 대신 과적합 위험도 조금 증가할 수 있음



</details>