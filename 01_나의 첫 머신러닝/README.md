# 정리

> 머신러닝의 지도학습에 대표적인 유형 중 하나 -> 분류(Classification)

1. **샘플(sample)** , **특성(feature)**

   - `bream_length` , `bream_weight` : 각 도미의 특성(길이, 무게)

   - 하나의 도미 정보: `[30, 600]` -> 특성 2개를 가진 샘플 하나

   <br/>

2. **레이블(label)** / **타겟(target)**
   
   ```py
   fish_target = [1]*35 + [0]*14
   ```

   - 도미는 `1` , 빙어는 `0` 으로 표현

   - 컴퓨터는 문자열보다 숫자를 좋아해서, 도미(`=1`), 빙어(`=0`) 처럼 이진 분류로 처리함

    <br/>

3. **리스트 내포(list comprehension)**

   ```py
   fish_data = [[l, w] for l, w in zip(length, weight)]
   ```

   - 각각 `[길이, 무게]` 형태로 묶은 2차원 배열을 만듦

   - `sklearn` 이 요구하는 입력 형식 (샘플 수 x 특성 수)

   <br/>

4. **K-최근접 이웃 분류기 (KNN - k-Nearest Neighbors)**

   ```py
   from sklearn.neighbors import KNeighborsClassifier
   ```

   - cf. `sklearn` **(사이킷런)**

      - 파이썬의 대표적인 머신러닝 라이브러리

      - 알고리즘, 전처리, 평가, 모델 저장/불러오기 등 포함

   - cf. `KNeighborsClassifier`

      - `sklearn.neighbors` 안에 있는 **k-최근접 이웃 분류 알고리즘 클래스**

      - 클래스를 만들고 -> 데이터를 주고 훈련시키고 -> 예측하게 만들 수 있음

      ```py
      from sklearn.neighbors import KNeighborsClassifier

      kn = KNeighborsClassifier()  # k=5가 기본값
      ```
   
   - 어떤 샘플이 주어졌을 때, 가장 가까운 k개의 샘플을 보고 투표해서 분류

   - `k = 5`일 경우 -> 주변의 5개 생선 중 도미가 더 많으면 도미로 분류

   ```py
   # 훈련 (train) 진행, 데이터와 정답을 모델에 알려줌
   kn = KNeighborsClassifier()
   kn.fit(fish_data, fish_target)
   ```

   ```py
   # 정확도(accuracy) 측정
   kn.score(fish_data, fish_target)
   ```

   - 훈련에 사용된 데이터를 그대로 예측하니, 정확도는 높게 나옴

    <br/>


5. **머신러닝 코드 흐름 정리**

   ```py
   from sklearn.neighbors import KNeighborsClassifier

   # 1. 모델 준비
   kn = KNeighborsClassifier(n_neighbors=5)

   # 2. 훈련 (학습)
   kn.fit(fish_data, fish_target)

   # 3. 평가 (정확도 확인)
   kn.score(fish_data, fish_target)

   # 4. 예측 (새로운 생선 분류)
   kn.predict([[30, 600]])
   ```
