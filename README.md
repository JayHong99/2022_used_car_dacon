# 2022_used_car_dacon


### 데이터 전처리 
- src/preprocess.py 참고
- 
```python
python preprocess.py
```


### 모델링
- load preprocessed data
- set parameter boundary
- modeling.py 참고
  - 모델 Random Forest 세팅
  - Bayeisan Optimization으로 최적 파라미터 탐색
  - SHAPely Values로 Model Explaining
```python
python main.py
```
