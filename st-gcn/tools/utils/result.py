import numpy as np
import torch
import torch.nn.functional as F   # 소프트맥스용
import sys

sys.path.append('/root/BoT-SORT/st-gcn')
from net.st_gcn import Model



#(base) root@1b5e3e640c55:~/st-gcn/resource/dataset/f_dataset/test# ls
#(base) root@1b5e3e640c55:~/st-gcn/resource/Data/ztid# ls



npy_path = '/root/BoT-SORT/st-gcn/resource/dataset/model_input/example1/train_data.npy'     # (N, C, T, V, M)
label_names = ['normal', 'falling']
ckpt_path = '/root/BoT-SORT/st-gcn/webcam_200.pt'
  # 파일명에 _ 대신 50
#ckpt_path = '/root/epoch200_model.pt'  #  ^l^l ^}   ^e ^w^p _  ^l^` ^k  50

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ST-GCN 모델 불러오기
model = Model(in_channels=3, num_class=2, edge_importance_weighting=True,
              graph_args={'layout': 'openpose', 'strategy': 'uniform'})
state_dict = torch.load(ckpt_path, map_location=device,weights_only=True)
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

data = np.load(npy_path)   # (N, C, T, V, M)

for i in range(data.shape[0]):
    input_tensor = torch.tensor(data[i]).unsqueeze(0).float().to(device)  # (1, C, T, V, M)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
        pred_label = label_names[pred]
    print(f"샘플 {i}: {pred_label}, probs={probs}")
