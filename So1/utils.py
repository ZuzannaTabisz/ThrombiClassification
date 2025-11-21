
import pyvips
import os
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def parse_images(folder):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    records = []

    for full_path in all_files:
        filename = os.path.basename(full_path)
        if '_' not in filename:
            continue  #skip malformed filenames

        parts = filename.replace('.jpg', '').split('_')
        if len(parts) < 2:
            continue 

        try:
           
            image_id = '_'.join(parts[:-1])
            instance_id = int(parts[-1])
            records.append({
                'image_path': full_path,
                'image_id': image_id,
                'instance_id': instance_id
            })
        except ValueError:
            
            print(f"Skip invalid file (cannot parse instance_id from '{parts[-1]}'): {filename}")
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(['image_id', 'instance_id']).reset_index(drop=True)
    return df


def merge_image_info(image_df, info_df):
    return image_df.merge(info_df, on='image_id', how='left').reset_index(drop=True)

def get_valid_transforms(cfg):
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5), 
        A.ShiftScaleRotate(
            shift_limit=0.02, 
            scale_limit=0.05, 
            rotate_limit=0, 
            border_mode=cv2.BORDER_REFLECT, 
            p=0.5
        ),
        A.RandomBrightnessContrast( 
            brightness_limit=0.05,
            contrast_limit=0.05, 
            p=0.3
        ),
        A.Normalize(mean=[0], std=[1], max_pixel_value=255.0),
        ToTensorV2(),
    ]) 


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0 
        self.avg = 0
        self.sum = 0 
        self.count = 0
    def update(self, val, n=1): 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 
