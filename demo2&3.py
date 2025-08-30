import sys
sys.path.append('.')
sys.path.append('./st-gcn')
sys.path.append('./EGRU')

#default
import argparse
import os
import os.path as osp
import time
import cv2
import torch
from loguru import logger

#ByteTracker
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
#from yolox.tracking_utils.visualize import plot_tracking
from yolox.utils.visualize import plot_tracking

#mmpose
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from collections import defaultdict, deque

#stgcn
import numpy as np
import torch
import torch.nn.functional as F
from net.st_gcn import Model

#EGRU
from net_egru.model import HighPerformanceEGRU
import pickle


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    # --- Demo & IO ---
    parser.add_argument("demo", default="image", help="demo type: image, video, webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")
    parser.add_argument("--save_result", action="store_true", help="save the inference result")

    # --- Exp & Model ---
    parser.add_argument("-f","--exp_file",default=None,type=str,help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device",default="gpu",type=str,help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16",dest="fp16",default=False,action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse",dest="fuse",default=False,action="store_true",help="Fuse conv and bn for testing.")
    parser.add_argument("--trt",dest="trt",default=False,action="store_true",help="Using TensorRT model for testing.")

    # --- Tracking ---
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="aspect ratio filter threshold")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # --- MMPose ---
    parser.add_argument("--mmpose-config", dest="mmpose_config",
                        default="mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py",
                        type=str, help="path to mmpose config file")
    parser.add_argument("--mmpose-checkpoint", dest="mmpose_checkpoint",
                        default="/root/.cache/mim/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
                        type=str, help="path to mmpose checkpoint file")

    # --- ST-GCN ---
    parser.add_argument("--stgcn-model", dest="stgcn_model", default="/root/ByteTrack/st-gcn/work_dir/recognition/kinetics_skeleton/uniform_aug/epoch200_model.pt",
                        type=str, help="path to pretrained ST-GCN model")
    # --- EGRU ---
    parser.add_argument("--egru-model", dest="egru_model", default="/root/ByteTrack/EGRU/egru_fall_detection_f7.pth",
                        type=str, help="path to pretrained EGRU model")

    parser.add_argument("--class-names", dest="class_names", default="class_names.txt",
                        type=str, help="path to action class names file")


    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    #{frame},{id},{x1},{y1},{w},{h},{s},dv, dtheta, aspect, daspect
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{dv},{dtheta},{aspect},{daspect}\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, feats in results:
            for tlwh, track_id, score,feat in zip(tlwhs, track_ids, scores, feats):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                dv, dtheta, aspect, daspect = feat
                line = save_format.format(
                    frame=frame_id, id=track_id,
                    x1=round(x1, 1), y1=round(y1, 1),
                    w=round(w, 1), h=round(h, 1),
                    s=round(score, 2),
                    dv=round(dv, 2), dtheta=round(dtheta, 2),
                    aspect=round(aspect, 2), daspect=round(daspect, 2)
                )
                f.write(line)
    logger.info('save results to {}'.format(filename))

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

#        ch = cv2.waitKey(0)
#        if ch == 27 or ch == ord("q") or ch == ord("Q"):
#            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def compute_iou(box1, box2):
    # box: (x, y, w, h)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union

def extract_features(tlwh, prev_state=None):
    x, y, w, h = tlwh
    cx, cy = x + w/2, y + h/2
    aspect = h / (w + 1e-6)
    area = w * h

    v, vx, vy, norm_v = 0, 0, 0, 0
    log_aspect, scale_ratio, area_ratio, iou = np.log(aspect+1e-6), 0, 0, 1.0

    if prev_state:
        dx=cx-prev_state["cx"]; 
        dy=cy-prev_state["cy"]; 
        v=np.sqrt(dx**2+dy**2)
        if v>1e-6: vx,vy=dx/v,dy/v
        norm_v=v/(h+1e-6)
        prev_area=prev_state["area"]
        scale_ratio=(area-prev_area)/(prev_area+1e-6)
        area_ratio=area/(prev_area+1e-6)
        iou=compute_iou((x,y,w,h),prev_state["bbox"])

    feat = np.array([
        norm_v,        # 정규화 속도
        vx, vy,        # 방향 단위벡터
        log_aspect,    # aspect 안정화
        scale_ratio,   # Δarea/area
        area_ratio,    # area 비율 (배율 관점)
        iou            # 프레임 IOU
    ], dtype=np.float32)

    new_state = {"cx": cx, "cy": cy, "v": v,
        "aspect": aspect, "area": area, "bbox": (x,y,w,h)
    }
    return feat, new_state




class BBoxFeatureCollector:
    def __init__(self, frame_len, save_dir="dataset", video_name=None):
        self.frame_len = frame_len
        self.save_dir = save_dir
        self.video_name = video_name
        os.makedirs(save_dir, exist_ok=True)

        self.buffers = defaultdict(lambda: deque(maxlen=frame_len))
        self.prev_states = {}
        self.samples = defaultdict(list)   # feature sequences
        self.frame_ids = defaultdict(list) # frame_id sequences


    def step(self, frame_id, tid, tlwh):
        prev_state = self.prev_states.get(tid)
        feat, new_state = extract_features(tlwh, prev_state)
        self.prev_states[tid] = new_state

        self.buffers[tid].append((frame_id,feat))

        if len(self.buffers[tid]) == self.frame_len:
            frames = list(map(lambda x: x[0], self.buffers[tid]))
            feats  = list(map(lambda x: x[1], self.buffers[tid]))

            seq = np.stack(feats, axis=0)  # (T,F)
            self.samples[tid].append(seq)
            self.frame_ids[tid].append(frames)

            self.buffers[tid].clear()


    def save(self, prefix="train"):
        save_folder = self.save_dir
        os.makedirs(save_folder, exist_ok=True)
        base_name = osp.splitext(osp.basename(self.video_name))[0]

        x_npy, meta = [], []
        for tid, seqs in self.samples.items():
            x = np.array(seqs, dtype=np.float32)  # (N, T, F)
            x_npy .append(x)
            for i, frames in enumerate(self.frame_ids[tid]):
                meta.append({"video": base_name, "tid": tid, "index": i, "frames": frames})

        if not x_npy:
            return


        x_npy = np.concatenate(x_npy, axis=0)  # (총 N, T, F)
        npy_path = os.path.join(save_folder, f"{prefix}_{base_name}.npy")
        pkl_path = os.path.join(save_folder, f"{prefix}_{base_name}.pkl")

        np.save(npy_path, x_npy)
        with open(pkl_path, "wb") as f:
            pickle.dump(meta, f)

class BBoxRecognizer:
    def __init__(self, args):
        self.model_path = args.egru_model
        self.frame_len = 30
        self.device = args.device or torch.device("cpu")
        self.class_names = ["normal", "falling"]

        self.model = HighPerformanceEGRU(
            in_dim=7,
            num_classes=2,
            hidden=256,
            num_layers=3,
            dropout=0.3
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.buffers = defaultdict(lambda: deque(maxlen=self.frame_len))
        self.tensor_buffer = torch.zeros((1, self.frame_len, 7), dtype=torch.float32, device=self.device)

    def step(self,frame_id, tid, bbox_feat):
        self.buffers[tid].append(bbox_feat)
        if len(self.buffers[tid]) == self.frame_len:
            seq = np.stack(self.buffers[tid], axis=0)
            self.tensor_buffer[0] = torch.from_numpy(seq).to(self.device)

            with torch.no_grad():
                logits = self.model(self.tensor_buffer)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
            self.buffers[tid].popleft()
            return tid, self.class_names[pred], probs.tolist()
        return None



def imageflow_demo(predictor, vis_folder, current_time,collector, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"VIDEO INFORM: WIDTH: {width}, HEIGHT: {height}, FPS: {fps}")

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    if args.save_result:
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    recognizer = BBoxRecognizer(args)
    action_dict,probs_dict = {},{}

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs,online_ids, online_scores = [],[],[]

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    x, y, w, h = tlwh
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh

                    if x < 0 or y < 0 or (x + w) > img_info['width'] or (y + h) > img_info['height']:
                        continue


                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        prev_state = collector.prev_states.get(tid)
                        feat, new_state = extract_features(tlwh, prev_state)
                        rec_result = recognizer.step(frame_id, tid, feat)


                        if rec_result is not None:
                            tid, pred_class, probs = rec_result
                            print(f"[Frame {frame_id}] Track {tid} -- {pred_class} {probs}")
                            action_dict[tid] = pred_class
                            probs_dict[tid] = probs

                        collector.prev_states[tid] = new_state
                        collector.step(frame_id, tid, tlwh)

                        if args.save_result:
                            base_info = f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}"
                            feat_str = ",".join([f"{v:.4f}" for v in feat])
                            results.append(f"{base_info},{feat_str}\n")

                timer.toc()
                online_im = plot_tracking_infer(
                    img_info['raw_img'], online_tlwhs, online_ids, scores=None, scales=None, frame_id=frame_id + 1, fps=1. / timer.average_time,
                    action_dict = action_dict, probs_dict = probs_dict
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']

            if args.save_result:
                vid_writer.write(online_im)

#            ch = cv2.waitKey(1)
#            if ch == 27 or ch == ord("q") or ch == ord("Q"):
#                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {vis_folder} {res_file}")

        if collector is not None:
            collector.save("train")
            logger.info(f"collector saved to /root/ByteTrack/{collector.save_dir}")


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # legs
]

def coco17_to_openpose18(keypoints_with_score):
    """
    Convert MMPose COCO17 (17x3) format → OpenPose18 (18x3).
    COCO17 order (MMPose):
      0:nose, 1:LEye, 2:REye, 3:LEar, 4:REar,
      5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow,
      9:LWrist, 10:RWrist, 11:LHip, 12:RHip,
      13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle

    OpenPose18 order:
      0:nose, 1:neck, 2:RShoulder, 3:RElbow, 4:RWrist,
      5:LShoulder, 6:LElbow, 7:LWrist,
      8:RHip, 9:RKnee, 10:RAnkle,
      11:LHip, 12:LKnee, 13:LAnkle,
      14:REye, 15:LEye, 16:REar, 17:LEar
    """

    kp = keypoints_with_score
    out = np.zeros((18, 3), dtype=np.float32)

    # Nose
    out[0] = kp[0]

    # Neck = average of LShoulder(5) + RShoulder(6)
    out[1, :2] = (kp[5, :2] + kp[6, :2]) / 2
    out[1, 2]   = (kp[5, 2] + kp[6, 2]) / 2  # avg score

    # Right arm
    out[2] = kp[6]   # RShoulder
    out[3] = kp[8]   # RElbow
    out[4] = kp[10]  # RWrist

    # Left arm
    out[5] = kp[5]   # LShoulder
    out[6] = kp[7]   # LElbow
    out[7] = kp[9]   # LWrist

    # Right leg
    out[8]  = kp[12]  # RHip
    out[9]  = kp[14]  # RKnee
    out[10] = kp[16]  # RAnkle

    # Left leg
    out[11] = kp[11]  # LHip
    out[12] = kp[13]  # LKnee
    out[13] = kp[15]  # LAnkle

    # Face
    out[14] = kp[2]  # REye
    out[15] = kp[1]  # LEye
    out[16] = kp[4]  # REar
    out[17] = kp[3]  # LEar

    return out



def draw_skeleton(img, frame_results, frame_id, score_thr=0.0):
    kps = frame_results.pred_instances.keypoints        # (N,K,2)
    scs = frame_results.pred_instances.keypoint_scores  # (N,K)

    for id  in range(len(kps)):
        points = kps[id]
        scores = scs[id]

        for i, (x, y) in enumerate(points):
            if scores[i] > score_thr:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

        for (i, j) in COCO_SKELETON:
            if scores[i] > score_thr and scores[j] > score_thr:
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[j]))
                cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    return img




class Action_Recognizer:
    def __init__(self,args):
        self.model = Model(
            in_channels=3,
            num_class= 2,
            edge_importance_weighting=True,
            graph_args={'layout': 'openpose', 'strategy': 'uniform'}
        )
        self.device = args.device
        self.class_names = ["normal","falling"]
        self.frame_len = 30
        self.V = 18
        self.C = 3
        self.buffers = defaultdict(lambda: deque(maxlen=self.frame_len))

        self.model.load_state_dict(torch.load(args.stgcn_model, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        #tensor buffer
        self.tensor_buffer = torch.zeros((1, self.C, self.frame_len, self.V, 1), dtype=torch.float32, device=self.device)

    def step(self, frame_id, tid, keypoints_with_score):
        flat_kpts = keypoints_with_score.reshape(-1).astype(np.float32)
        cv = flat_kpts.reshape(self.V, self.C).T
        self.buffers[tid].append(cv)

        if len(self.buffers[tid]) == self.frame_len:
            np_data = np.stack(self.buffers[tid], axis=1)   # (C,T,V)
            self.tensor_buffer[0, :, :, :, 0] = torch.from_numpy(np_data)

            with torch.no_grad():
                output = self.model(self.tensor_buffer)           # (1,num_class)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)

            self.buffers[tid].popleft()
            return tid, self.class_names[pred], probs.tolist()
        return None

    def print_buffers(self):
        print("=== BUFFER CONTENTS ===")
        for tid, buf in self.buffers.items():
            print(f"TID {tid}: {len(buf)}")


def plot_tracking_infer(image, tlwhs, obj_ids, scores, scales, frame_id=0, fps=0., action_dict=None,probs_dict=None,egru_dict=None, stgcn_dict=None):
    im = image.copy()
    text_scale = 1.0
    text_thickness = 2
    line_thickness = 3

    for i, (tlwh, obj_id) in enumerate(zip(tlwhs, obj_ids)):
        x1, y1, w, h = tlwh
        x2, y2 = int(x1 + w), int(y1 + h)
        x1, y1 = int(x1), int(y1)

        action_str = action_dict.get(obj_id, "loading...") if action_dict else "loading..."
        color = (0, 0, 255) if action_str == "falling" else (0, 255, 0)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=line_thickness)

        prob_str = ""
        if probs_dict and obj_id in probs_dict:
            p = probs_dict[obj_id]
            prob_str = f" {p[1]:.2f}"

        score_str = f" conf:{scores[i]:.2f}" if scores else "None"
        scales_str = f" scales:{scales[i]:.2f}" if scales else "None"

        egru_str, stgcn_str = "", ""
        if egru_dict and obj_id in egru_dict:
            egru_str = f" egru={egru_dict[obj_id]}"
        if stgcn_dict and obj_id in stgcn_dict:
            stgcn_str = f" stgcn={stgcn_dict[obj_id]}"


        cv2.putText(im,
                    f"ID {obj_id}: {action_str}{egru_str}{stgcn_str}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (255, 255, 255),
                    thickness=text_thickness)

    cv2.putText(im,
                f"frame: {frame_id} fps: {fps:.2f}",
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),
                thickness=text_thickness)

    return im



def imageflow_demo_gpt(predictor, vis_folder, current_time, collector, args):
    cap = cv2.VideoCapture(args.path if args.demo == "bms" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"VIDEO INFORM:  WIDTH: {width}, HEIGHT: {height},FPS: {fps}")

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1] if args.demo == "bms" else "camera.mp4")
    logger.info(f"VIDEO WILL BE SAVED ... {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    #============bms================
    tracker = BYTETracker(args, frame_rate=30)
    mmpose = init_model(args.mmpose_config, args.mmpose_checkpoint, device=args.device)
    bbox_recognizer = BBoxRecognizer(args)
    stgcn_recognizer = Action_Recognizer(args)
    #===============================
    timer = Timer()
    frame_id = 0
    egru_dict,stgcn_dict,results = {},{},{}

    while True:
        if frame_id % 20 == 0:
            logger.info(
                'Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time))
            )

        ret_val, frame = cap.read()
        if not ret_val:
            print("fail")
            break

        outputs, img_info = predictor.inference(frame, timer)

        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

            bboxes, tids = [], []
            online_tlwhs, online_ids, online_scores, online_scales = [], [], [], []
            action_dict = {}
            img_h, img_w = img_info['raw_img'].shape[:2]

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                x, y, w, h = tlwh
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh

                if x < 0 or y < 0 or (x + w) > img_info['width'] or (y + h) > img_info['height']:
                    continue

                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    x1, y1, w, h = map(int, tlwh)
                    bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                    bboxes.append(np.array(bbox_xyxy, dtype=np.float32)) #t.score>0.5:
                    tids.append(tid)
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    # ========= BBoxRecognizer (EGRU) ===========
                    prev_state = collector.prev_states.get(tid)
                    feat, new_state = extract_features(tlwh, prev_state)
                    rec_result = bbox_recognizer.step(frame_id, tid, feat)

                    if rec_result is not None:
                        tid, pred_class, probs = rec_result
                        probs_bbox = np.array(probs)
                        pred_bbox = np.argmax(probs_bbox)
                    else:
                        probs_bbox = np.array([0.5, 0.5])  # fallback
                        pred_bbox = 0

                    collector.prev_states[tid] = new_state
                    egru_dict[tid] = probs_bbox

                    #collector.step(frame_id, tid, tlwh)
                    # ===========================================


            # ========= ST-GCN =========
            if len(bboxes) > 0:
                batch_results = inference_topdown(mmpose, img_info['raw_img'], bboxes)
                ds  = merge_data_samples(batch_results)

                for (tid, pose_result,(x1, y1, box_w, box_h)) in zip(tids, batch_results, [map(int, tlwh) for tlwh in online_tlwhs]):
                    if pose_result and pose_result.pred_instances.keypoints.shape[1] > 0:
                        raw_kp = pose_result.pred_instances.keypoints[0]
                        raw_score = pose_result.pred_instances.keypoint_scores[0]

                        thr = 0.7  # confidence threshold
                        kp = raw_kp.copy()
                        kp[:, 0] = raw_kp[:, 0] / img_info['width']
                        kp[:, 1] = raw_kp[:, 1] / img_info['height']

                        keypoints_with_score = np.concatenate([kp, raw_score[:, None]], axis=1)
#                        low_conf = raw_score < thr
#                        keypoints_with_score[low_conf, :] = 0.0
                        keypoints_with_score = coco17_to_openpose18(keypoints_with_score)

                    else:
                        keypoints_with_score = np.zeros((18, 3), dtype=np.float32)

                    result = stgcn_recognizer.step(frame_id, tid, keypoints_with_score)
                    if result:
                        tid, action, probs_pose = result
                        probs_pose = np.array(probs_pose)
                        pred_pose = np.argmax(probs_pose)

                        #action_dict[tid] = action
                        #print(f"[ST-GCN] {tid}: {action} {tid} {probs}")
                    else:
                        probs_pose = np.array([0.5, 0.5])
                        pred_pose = 0  # fallback → normal

                    stgcn_dict[tid] = probs_pose


                    probs_bbox = egru_dict.get(tid, np.array([0.5, 0.5]))
                    # ========= Fusion(alpha, beta) =========
                    alpha ,beta=0.4, 0.6
                    final_probs = alpha * probs_bbox + beta * probs_pose
                    final_pred = np.argmax(final_probs)
                    final_action = stgcn_recognizer.class_names[final_pred]

                    #if pred_bbox == 1 and pred_pose == 1:
                    #    final_action = "falling"
                    #else:
                    #    final_action = "normal"

                    action_dict[tid] = final_action

                    print(f"{frame_id} : Track {tid}: {final_action} "
                          f"(egru={np.round(probs_bbox,2)}, stgcn={np.round(probs_pose,2)}, fused={np.round(final_probs,2)})")

                    #print(f"{frame_id} :  Track {tid}: {final_action} (egru={probs_bbox}, stgcn={probs_pose}, fused={final_probs})")


                draw_img = draw_skeleton(img_info['raw_img'], ds, frame_id, score_thr=0.0)
            else:
                draw_img = img_info['raw_img'].copy()
            timer.toc()

        else:
            timer.toc()
            draw_img = img_info['raw_img']



        online_im = plot_tracking_infer(
                draw_img, online_tlwhs, online_ids,online_scores,online_scales,
                frame_id=frame_id + 1,
                fps=1. / max(1e-5, timer.average_time),
                action_dict=action_dict)



        if args.save_result:
            vid_writer.write(online_im)

#        ch = cv2.waitKey(1)
#        if ch in [27, ord("q"), ord("Q")]:
#            break

        frame_id += 1

    cap.release()
    vid_writer.release()
    stgcn_recognizer.print_buffers()
    logger.info(f"save results to {save_path}")




def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = "temp"

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    collector = BBoxFeatureCollector(frame_len=30,video_name=args.path)


    current_time = time.localtime()
    if args.demo == "video":
        imageflow_demo(predictor, vis_folder, current_time, collector, args)
    elif args.demo == "bms" or args.demo == "webcam":
        imageflow_demo_gpt(predictor, vis_folder, current_time,collector, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)



#def extract_features(tlwh, prev_state=None):
#    """
#    tlwh: [x, y, w, h]
#    prev_state: {"cx":..., "cy":..., "vx":..., "vy":...}
#    """
#    x, y, w, h = tlwh
#    cx, cy = x + w/2, y + h/2
#    aspect = h / (w + 1e-6)
#
#    vx, vy, speed, accel = 0, 0, 0, 0
#    if prev_state:
#        vx = cx - prev_state["cx"]
#        vy = cy - prev_state["cy"]
#        speed = np.sqrt(vx**2 + vy**2)
#
#        if "vx" in prev_state:
#            dvx = vx - prev_state["vx"]
#            dvy = vy - prev_state["vy"]
#            accel = np.sqrt(dvx**2 + dvy**2)
#
#    feat = np.array([cx, cy, vx, vy, speed, accel, aspect], dtype=np.float32)
#    new_state = {"cx": cx, "cy": cy, "vx": vx, "vy": vy}
#    return feat, new_state

#class BBoxRecognizer:
#    def __init__(self, model, frame_len=30, class_names=None):
#        self.model = model
#        self.frame_len = frame_len
#        self.class_names = class_names or ["normal", "falling"]

#        self.buffers = defaultdict(lambda: deque(maxlen=frame_len))

#    def step(self, frame_id, tid, bbox_feat):
#        """
#        bbox_feat: (7,) numpy array
#                   [cx, cy, vx, vy, speed, a, aspect]
#        """
#        self.buffers[tid].append(bbox_feat)

#        if len(self.buffers[tid]) == self.frame_len:
#            seq = np.stack(self.buffers[tid], axis=0)  # (T,F)
#            tensor_in = torch.from_numpy(seq).unsqueeze(0).float()  # (1,T,F)

#            with torch.no_grad():
#                logits = self.model(tensor_in)          # (1,num_classes)
#                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
#                pred = np.argmax(probs)

            # stride=1 sliding
#            self.buffers[tid].popleft()
 
#            return tid, self.class_names[pred], probs.tolist()
#        return None


#def extract_features(tlwh, prev_state=None):
#    x, y, w, h = tlwh
#    cx, cy = x + w/2, y + h/2
#    aspect = h / (w + 1e-6) 

#    dv,speed_ratio, dtheta, daspect = 0, 0, 0, 0
#    v, theta = 0, None

#    if prev_state and "cx" in prev_state:
#        dx = cx - prev_state["cx"]
#        dy = cy - prev_state["cy"]
#        v = np.sqrt(dx**2 + dy**2)
#        if "v" in prev_state:
#            dv = v - prev_state["v"]
            #speed_ratio = dv / (prev_state["v"] + 1e-6)

#        theta = np.arctan2(dy, dx + 1e-6)
#        if prev_state.get("theta") is not None:
#            dtheta = theta - prev_state["theta"]
#            dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
#        if "aspect" in prev_state:
#            daspect = aspect - prev_state["aspect"] 

#    feat = np.array([dv, dtheta, aspect, daspect], dtype=np.float32)
#    new_state = {"cx": cx, "cy": cy, "v": v, "theta": theta, "aspect": aspect}
#    return feat, new_state
