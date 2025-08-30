
이 저장소는 진명아이엔씨 현장실습 기간 동안 연구한 낙상 탐지 모델을 구현한 코드입니다.

## 0. 폴더 설명
<pre>
project
|-- mim: pretrained mmpose model
|-- mmpose
|-- net egru
|   |--pretrained:pretraiend model
|   |--model:egru.net
|-- stgcn
|   |--feeder:preprocess
|   |--tools: preprocess 
|-- yolox
|-- bytetrack
|-- bot-sort
</pre>

## 1. 초기 세팅 
서버 환경 RTX A6000 GPU 0,1번 총 두 개를 사용했으며 회사내 GPU서버 , 교내 서버 등을 이용하였습니다.
cuda_11.8 버전에서 돌렸으며 환경이 동일한 경우 requirements.txt를 참고하여 환경설정 하면 되겠습니다.

detection & tracking 모델 : BoT-SORT, ByteTrack, yolox-mot17  
pose estimation 모델: openpose,mmpose  
inference 모델 : stgcn, egru 



## 🚀 demo 1) BoT-SORT + MMPose + ST-GCN
<img width="1208" height="671" alt="image" src="https://github.com/user-attachments/assets/6cdd23b0-def6-40db-9c0b-02083645ed06" />

1. BoT-SORT(re-id 적용 및 track_id별 bbox 정보 추출) 
2. MMPose(프레임 track별 스켈레톤 추출)
3. 3.st-gcn 활용한 트랙별 falling inference 실행

## 🚀 demo 2) ByteTrack + extract feature +  EGRU

https://github.com/user-attachments/assets/e09cf97f-1c34-4e24-9f08-8138ea0170f8



https://github.com/user-attachments/assets/a406661b-7f88-4295-bcb1-4e6fb0067971




1. ByteTrack(track_id별 bbox 정보 추출)
2. compute bbox feature
3. egru 활용한 트랙별 falling inference 실행
## 🚀 demo 3) ByteTrack + Fusion(EGRU/ST-GCN)



https://github.com/user-attachments/assets/46713cb5-302f-407c-8430-9b410d892149


1. ByteTrack(track_id별 bbox 정보수집) 
2. bbox 정보 추출 및 피처 계산  
3. MMPose(프레임 track별 스켈레톤 추출)
4. Fusion egru,stgcn falling inference 


