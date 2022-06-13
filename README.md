# 2022_used_car_dacon


### 데이터 전처리 
```python
python preprocess.py
```
- src/preprocess.py 참고



### 모델링
```python
python main.py
```
- load preprocessed data
- set parameter boundary
- modeling.py 참고
  - 모델 Random Forest 세팅
  - Bayeisan Optimization으로 최적 파라미터 탐색
  - SHAPely Values로 Model Explaining
