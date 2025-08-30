# Real-time Fall Detection

본 저장소는 다양한 방식의 낙상 탐지 구현을 정리합니다.

------------------------------------------
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

코드에 쓰인 tracking 모델은 : BoT-SORT, ByteTrack, yolox-mot17 | pose estimation 모델은: openpose,mmpose |inference 모델은 : stgcn, gru 입니다.



## 🚀 demo 1) BoT-SORT + MMPose + ST-GCN
1.BoT-SORT(re-id 적용 및 track_id별 bbox 정보 추출) 
2. MMPose(프레임 track별 스켈레톤 추출)
3. 3.st-gcn 활용한 트랙별 falling inference 실행

## 🚀 demo 2) ByteTrack + extract feature +  EGRU
1. ByteTrack(track_id별 bbox 정보수집)
2. bbox 정보 추출 및 피처 계산
3. egru 활용한 트랙별 falling inference 실행
## 🚀 demo 3) ByteTrack + Fusion(EGRU/ST-GCN)
1. ByteTrack(track_id별 bbox 정보수집) 
2. bbox 정보 추출 및 피처 계산  
3. MMPose(프레임 track별 스켈레톤 추출)
4. egru,stgcn fusion falling inference 


