import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class COCODataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.file_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(self.img_dir)]
        self.cache_path = Path('/home/mp/work/track_anything/.cache_dataset')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, f'{file_name}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{file_name}.png')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not len(mask_contours):
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]
            masks = [np.zeros(image.shape[:2], dtype = "uint8")]
        else:
            bboxes = []
            masks = []

            mask_contour_min_area = 100
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) < mask_contour_min_area:
                    continue
                bboxes.append(cv2.boundingRect(mask_contour))

                mask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.drawContours(mask, [mask_contour], 0, 1, -1)
                masks.append(mask)

            bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
        
        embeding = self.cache_path / f'{file_name}.npy'
        if embeding.is_file():
            embeding = np.load(embeding)
            embeding = torch.tensor(embeding)

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        bboxes = torch.tensor(bboxes)
        masks = torch.tensor(masks).float()
        return image, bboxes, masks, embeding


def collate_fn(batch):
    images, bboxes, masks, embedings = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks, embedings


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    
    train = COCODataset(img_dir=cfg.dataset.train.img_dir,
                        mask_dir=cfg.dataset.train.mask_dir,
                        transform=transform)
    val = COCODataset(img_dir=cfg.dataset.val.img_dir,
                      mask_dir=cfg.dataset.val.mask_dir,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
