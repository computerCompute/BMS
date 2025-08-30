# Real-time Fall Detection

본 저장소는 다양한 방식의 낙상 탐지 구현을 정리합니다.

------------------------------------------
## 0. 폴더 설명
<pre>
project
|-- mim: pretrained mmpose model
|-- mmpose
│ 
│   ├──saved_pkl:전처리 데이터 저장
│── predict_video:낙상 결과 추론 코드(시각화)
│── stgcn:
│   ├──config:custom yaml
│   ├──model:학습 모델
│   ├──net:stgcn.py
│── webcam : 웹캠 실시간 낙상추론 코드(skeleton 시각화)
</pre>

## 🚀 Case 1) BoT-SORT + MMPose + ST-GCN

## 🚀 Case 2) ByteTrack + EGRU

## 🚀 Case 3) ByteTrack + Fusion(EGRU/ST-GCN)



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
