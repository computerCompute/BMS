import os
import sys
import sys
import json
import argparse
from pathlib import Path

sys.path.append('/root/BoT-SORT/st-gcn/tools/utils')
#print(sys.path)
from json_pack import json_pack

def preprocess_pack(input_dir, output_dir, frame_width, frame_height, label, label_index, frames_per_sample):
    os.makedirs(output_dir, exist_ok=True)

    example_folders = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    ]


    total_count = 0

    for example_folder in example_folders:
        example_folder_path = Path(example_folder)
        json_files = sorted(list(example_folder_path.glob('*.json')))
        num_samples = len(json_files) // frames_per_sample
        print("processing ... {example_folder}")
        for i in range(num_samples):
            batch = json_files[i*frames_per_sample : (i+1)*frames_per_sample]
            if len(batch) < frames_per_sample:
                continue
            save_file_name = f"{label}_{example_folder_path.name}_{i+1:03d}.json"
            video_info = json_pack(batch, frame_width, frame_height, label, label_index)
            save_path = os.path.join(output_dir, save_file_name)
            with open(save_path, 'w') as f:
                json.dump(video_info, f)
            total_count += 1
    print(f"{total_count} samples saved for label '{label}' in {output_dir}")

    print(f"input_dir={input_dir}, output_dir={output_dir}, label={label}, label_index={label_index}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--frame_width', type=int, default=1920)
    parser.add_argument('--frame_height', type=int, default=1080)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--label_index', type=int, required=True)
    parser.add_argument('--frames_per_sample', type=int, required = True)
    args = parser.parse_args()

    preprocess_pack(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        label=args.label,
        label_index=args.label_index,
        frames_per_sample=args.frames_per_sample
    )



'''
-- command --

python preprocess.py \
  --input_dir /root/st-gcn/resource/Data/test \
  --output_dir /root/st-gcn/resource/Dataset/f_json_pack/test  \
  --frame_width 1920 \
  --frame_height 1080 \
  --label falling \
  --label_index 1 \
  --frames_per_sample 30
'''


