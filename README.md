# FALL_DETECTION(BMS)
case1) BoT-SORT + mmpose + stgcn \n
case2) ByteTrack + egru \n 
case3) ByteTrack + fusion(egru/stgcn) \n


## 0. 폴더 설명
<pre>
project
├── data: 학습 데이터
│   ├── falling_video: 낙상 영상
│   ├── normal_video: 보행 영상
│── dataset:전처리 데이터 및 코드
│   ├──saved_pkl:전처리 데이터 저장
│── predict_video:낙상 결과 추론 코드(시각화)
│── stgcn:
│   ├──config:custom yaml
│   ├──model:학습 모델
│   ├──net:stgcn.py
│── webcam : 웹캠 실시간 낙상추론 코드(skeleton 시각화)
</pre>
