import json
import os
import numpy as np
import pandas as pd


DATA_DIR = './data'
FOLDERS = ['Normal', 'Mild', 'Moderate', 'Severe']
LABEL_MAP = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

HIP_IDX = 18
R_ANKLE_IDX = 15
L_ANKLE_IDX = 16
R_WRIST_IDX = 10 
L_WRIST_IDX = 9

all_results = []

print("Starting extraction: Step Length, Arm Swing, and Stability Index...")

for folder in FOLDERS:
    folder_path = os.path.join(DATA_DIR, folder)
    label = LABEL_MAP[folder]
    
    if not os.path.exists(folder_path):
        print(f"Skipping {folder}: Folder not found.")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            video_step_lengths = []
            video_arm_swings = []

            for frame in data:
        
                keypoints = np.array(frame['keypoints']).reshape(-1, 3)
                
               
                l_ankle = keypoints[L_ANKLE_IDX][:2]
                r_ankle = keypoints[R_ANKLE_IDX][:2]
                video_step_lengths.append(np.linalg.norm(l_ankle - r_ankle))
                
               
                r_wrist = keypoints[R_WRIST_IDX][:2]
                hip = keypoints[HIP_IDX][:2]
                video_arm_swings.append(np.linalg.norm(r_wrist - hip))
            
            if video_step_lengths:
                avg_step = np.mean(video_step_lengths)
                std_step = np.std(video_step_lengths)
                avg_arm = np.mean(video_arm_swings)
                
                
                stability_index = avg_step / (std_step + 1e-6)

                all_results.append({
                    'avg_step_length': avg_step,
                    'std_step_length': std_step,
                    'avg_arm_swing': avg_arm,
                    'stability_index': stability_index, 
                    'label': label
                })

df = pd.DataFrame(all_results)
df.to_csv('mmu_pd_features.csv', index=False)

print("-" * 30)
print(f"Success! Dataset saved: mmu_pd_features.csv")
print(f"Total videos processed: {len(df)}")
print("-" * 30)