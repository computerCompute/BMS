import sys
sys.path.append('/root/BoT-SORT/st-gcn')
sys.path.append('.')


import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
from loguru import logger
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from termcolor import cprint
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn.functional as F
from net.st_gcn import Model

import warnings
warnings.filterwarnings("ignore")




def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument("--stgcn-model", dest="stgcn_model", default="st-gcn/work_dir/recognition/kinetics_skeleton/uniform_aug/epoch200_model.pt", type=str, help="path to pretrained ST-GCN model")
    parser.add_argument("--class-names", dest="class_names", default="class_names.txt", type=str, help="path to action class names file")


    return parser

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


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
#        img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

def init_mmpose():
    config = 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
    checkpoint = '/root/.cache/mim/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
    model = init_model(config, checkpoint, device='cuda:1')
    return model

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
    def __init__(self, model, device, class_names, frame_len, V, C,args):
        self.model = Model(
            in_channels=3,
            num_class= 2,
            edge_importance_weighting=True,
            graph_args={'layout': 'openpose', 'strategy': 'uniform'}
        )
        self.device = device
        self.class_names = class_names
        self.frame_len = frame_len
        self.V = V
        self.C = C
        self.buffers = defaultdict(lambda: deque(maxlen=frame_len))

        self.model.load_state_dict(torch.load(args.stgcn_model, map_location=device))
        self.model = self.model.to(self.device).eval()
        #tensor buffer
        self.tensor_buffer = torch.zeros((1, C, frame_len, V, 1), dtype=torch.float32, device=device)

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


def plot_tracking(image, tlwhs, obj_ids, scores, scales, frame_id=0, fps=0., action_dict=None):
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

        score_str = f" conf:{scores[i]:.2f}" if scores else "None"
        scales_str = f" scales:{scales[i]:.2f}" if scales else "None"

        cv2.putText(im,
                    f"ID {obj_id}: {action_str}{score_str}{scales_str}",
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



def imageflow_demo_gpt(predictor, vis_folder, current_time, args):

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"VIDEO INFORM:  WIDTH: {width}, HEIGHT: {height},FPS: {fps}")
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1] if args.demo == "video" else "camera.mp4")
    logger.info(f"VIDEO WILL BE SAVED ... {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    #============BMS================
    tracker = BoTSORT(args, frame_rate=args.fps)
    mmpose = init_mmpose()
    class_names = ["normal", "falling"]
    recognizer = Action_Recognizer(args.stgcn_model, args.device, class_names, 30, 18, 3, args)
    #===============================
    timer = Timer()
    frame_id = 0
    results = []

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
        scale = min(
            exp.test_size[0] / float(img_info['height']),
            exp.test_size[1] / float(img_info['width'])
        )

        #online_tlwhs, online_ids, online_scores, online_scales = [], [], [], []
        #action_dict = {}

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            online_targets = tracker.update(detections, img_info["raw_img"])


            bboxes, tids = [], []
            online_tlwhs, online_ids, online_scores, online_scales = [], [], [], []
            action_dict = {}
            img_h, img_w = img_info['raw_img'].shape[:2]

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh

                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    x1, y1, w, h = map(int, tlwh)
                    bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                    bboxes.append(np.array(bbox_xyxy, dtype=np.float32))
                    tids.append(tid)
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_scales.append(h / w)
#                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical and t.score>0.5:
#                       x1, y1, w, h = map(int, tlwh)
#                       x2, y2 = x1 + w, y1 + h
#                       if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h:
#                           bbox_xyxy = [x1, y1, x2, y2]
#                           bboxes.append(np.array(bbox_xyxy, dtype=np.float32))
#                           tids.append(tid)
#                           online_tlwhs.append(tlwh)
#                           online_ids.append(tid)
#                           online_scores.append(t.score)
#                           online_scales.append(h / w)

            if len(bboxes) > 0:
                print(f" len box : {len(bboxes)}")
                batch_results = inference_topdown(mmpose, img_info['raw_img'], bboxes)
                ds  = merge_data_samples(batch_results)

                for (tid, pose_result,(x1, y1, box_w, box_h)) in zip(tids, batch_results, [map(int, tlwh) for tlwh in online_tlwhs]):
                    if pose_result and pose_result.pred_instances.keypoints.shape[1] > 0:
                        raw_kp = pose_result.pred_instances.keypoints[0]
                        raw_score = pose_result.pred_instances.keypoint_scores[0]

                        kp = raw_kp.copy()
                        kp[:, 0] = raw_kp[:, 0] / img_info['width']
                        kp[:, 1] = raw_kp[:, 1] / img_info['height']

                        keypoints_with_score = np.concatenate([kp, raw_score[:, None]], axis=1)
                        keypoints_with_score = coco17_to_openpose18(keypoints_with_score)

                    else:
                        keypoints_with_score = np.zeros((18, 3), dtype=np.float32)

                    result = recognizer.step(frame_id, tid, keypoints_with_score)
                    if result:
                        tid, action, probs = result
                        action_dict[tid] = action
                        print(f"[ST-GCN] {tid}: {action} {tid} {probs}")
                    else:
                        action_dict[tid] = "loading..."

                draw_img = draw_skeleton(img_info['raw_img'], ds, frame_id, score_thr=0.0)
            else:
                draw_img = img_info['raw_img'].copy()
            timer.toc()

        else:
            timer.toc()
            online_im = img_info['raw_img']



        online_im = plot_tracking(
                draw_img, online_tlwhs, online_ids,online_scores,online_scales,
                frame_id=frame_id + 1,
                fps=1. / max(1e-5, timer.average_time),
                action_dict=action_dict)



        if args.save_result:
            vid_writer.write(online_im)

        ch = cv2.waitKey(1)
        if ch in [27, ord("q"), ord("Q")]:
            break

        frame_id += 1

    cap.release()
    vid_writer.release()
    recognizer.print_buffers()
    logger.info(f"save results to {save_path}")



def main(exp, args):
    # ----- Basic settings -----
    output_dir = osp.join(exp.output_dir, args.experiment_name or exp.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    logger.info("Args: {}".format(args))

    # ----- Detector (YOLOX) -----
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info("Detector checkpoint loaded.")

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, None, None, args.device, args.fp16)
    current_time = time.localtime()

    # ----- Run demo -----
    if args.demo == "video":
        imageflow_demo_gpt(predictor, vis_folder, current_time,  args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)

