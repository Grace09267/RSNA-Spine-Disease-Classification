# RSNA Spine Disease Classification

## 📌 프로젝트 개요
본 프로젝트는 RSNA Spine 데이터를 활용하여 척추 디스크 질환을 분류하는 모델을 개발
CNN, LSTM, Transformer 기반 구조를 비교 분석하는 것을 목표로 함
특히, sequence 모델이 실제로 디스크 간 관계를 학습하는지 검증하기 위해 sequence shuffle 실험을 수행

## 📌 목적
특정 slice의 병변의 위치를 선택하면 다른 slice로 좌표가 전파되고, 다른 척추와의 연관성을 살펴보며 질환을 분류하는 알고리즘을 개발하고자 함

## 📌 모델 구조
1. Experiment 1
- CNN + LSTM
2. Experiment 2
- CNN + LSTM + Coordinate-guided Attention
3. Experiment 3
- CNN + Transformer

## 📌 핵심 결과
- LSTM 및 Transformer 모델 모두 sequence 순서를 무작위로 섞어도 성능 저하가 거의 발생하지 않음
- 이는 모델이 디스크 간 관계(sequence dependency)를 학습하지 않았음을 의미함
- 각 디스크 patch만으로도 충분한 classification 성능이 확보됨
###  성능 비교
|        Model       | Original | Shuffled |
|--------------------|----------|----------|
|     CNN + LSTM     |   0.746  |   0.739  |
| CNN + LSTM + Coord |   0.798  |   0.789  |
| CNN + Transformer  |   0.919  |   0.919  |

## 📌 Insight
- 본 문제는 sequence modeling 문제가 아니라 independent classification 문제에 가까움
- Transformer의 성능 향상은 sequence 학습이 아니라 feature aggregation 효과로 해석됨
- 좌표 정보(coord)는 sequence 관계 학습이 아닌 위치 bias로 작용함

## 📌 데이터 처리
- Registration 기반으로 디스크 위치 정렬
- GT의 좌표 전파 
- 각 디스크 중심으로 ROI crop 수행
- slice 단위 데이터 생성

## 📌 실행방법
pip install -r requirements.txt
전처리(pre_process_all.py -> build_roi_dataset.py -> convert_pt_to_npy.py)
학습
- experiment1.py
- experiment2.py
- experiment3.py
history(notebooks/analysis.ipynb)

⚠️ 주의사항
- 데이터셋은 포함되어 있지 않습니다.(https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- outputs/ 폴더의 데이터는 직접 생성해야 합니다.
