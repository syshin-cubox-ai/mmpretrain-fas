import glob
import json
import math
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tvf
import tqdm

src_path = 'D:/data/OULU_processed_one_frame'
dst_path = 'D:/data/mmpretrain_custom'
subset = 'train'
# subset = 'val'
img_size = 256


def resized_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, size: int) -> np.ndarray:
    x1 = np.clip(x1, 0, img.shape[1])
    y1 = np.clip(y1, 0, img.shape[0])
    x2 = np.clip(x2, 0, img.shape[1])
    y2 = np.clip(y2, 0, img.shape[0])
    w = x2 - x1
    h = y2 - y1

    # Margin to crop to square
    if h > w:
        left_margin, right_margin = math.floor((h - w) / 2), math.ceil((h - w) / 2)
        x1 -= left_margin
        x2 += right_margin
    elif w > h:
        up_margin, down_margin = math.floor((w - h) / 2), math.ceil((w - h) / 2)
        y1 -= up_margin
        y2 += down_margin

    # Pad if the coordinates exceed the image range
    crop_img = tvf.crop(torch.from_numpy(img).permute(2, 0, 1), y1, x1, y2 - y1, x2 - x1).permute(1, 2, 0).numpy()
    assert crop_img.shape[0] == crop_img.shape[1]

    scale = size / crop_img.shape[0]
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized_crop_img = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized_crop_img


# Check datdaset
img_paths = sorted(glob.glob(os.path.join(src_path, '**/*.jpg'), recursive=True))
label_paths = sorted(glob.glob(os.path.join(src_path, '**/*.json'), recursive=True))
for img_path, label_path in zip(img_paths, label_paths, strict=True):
    assert os.path.splitext(img_path)[0] == os.path.splitext(label_path)[0], 'Mismatch image and label pair.'

os.makedirs(os.path.join(dst_path, subset, 'fake'), exist_ok=True)
os.makedirs(os.path.join(dst_path, subset, 'live'), exist_ok=True)
for img_path, label_path in tqdm.tqdm(zip(img_paths, label_paths, strict=True), 'Process', len(img_paths)):
    # Load image and label
    img = cv2.imread(img_path)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    # Crop and resize image to desired size.
    img = resized_crop(img, label['x1'], label['y1'], label['x2'], label['y2'], img_size)

    # Save image according to label
    if label['label'] == 0:
        subfolder = 'fake'
    elif label['label'] == 1:
        subfolder = 'live'
    else:
        raise ValueError(f'Wrong label: {label["label"]}')
    assert not os.path.exists(os.path.join(dst_path, subset, subfolder, os.path.basename(img_path)))
    cv2.imwrite(os.path.join(dst_path, subset, subfolder, os.path.basename(img_path)),
                img, [cv2.IMWRITE_JPEG_QUALITY, 100])
