import os
import json
from pathlib import Path


# len(json_file_list) -> dic(video_info)
def json_pack(json_file_list, frame_width, frame_height, label='unknown', label_index=-1):
    sequence_info = []
    for idx, json_path in enumerate(json_file_list):
        frame_data = {'frame_index': idx}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                score += [keypoints[i + 2]]
            skeletons.append({'pose': coordinates, 'score': score})
        frame_data['skeleton'] = skeletons
        sequence_info.append(frame_data)
    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    return video_info



def json_pack2(json_file_list, frame_width, frame_height, label='unknown', label_index=-1):
    sequence_info = []
    for idx, json_path in enumerate(json_file_list):
        frame_data = {'frame_index': idx}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i], keypoints[i + 1]]
                score += [keypoints[i + 2]]
            skeletons.append({'pose': coordinates, 'score': score})
        frame_data['skeleton'] = skeletons
        sequence_info.append(frame_data)
    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    return video_info
