import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def plot_mask_on_img(img, mask):
    # img: h,w,c
    # masks: h,w
    img = img.astype(np.int64, copy=True)
    color = [255,0,0]
    img[mask>=1] += np.array(color)
    img[mask>0] //= 2
    img = img.astype(np.uint8)
    return img


class COCODataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.file_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(self.img_dir)]

        # self.image_ids = list(self.coco.imgs.keys())

        # # Filter out image_ids without any annotations
        # self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

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
            bboxes = [cv2.boundingRect(cnt) for cnt in mask_contours]
            bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
            masks = []

            mask_contour_min_area = 100 
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) < mask_contour_min_area:
                    continue

                mask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.drawContours(mask, [mask_contour], 0, 1, -1)
                
                masks.append(mask)
        
        # img = plot_mask_on_img(image, masks[0])
        # img = cv2.rectangle(img, (bboxes[0][0][0], bboxes[0][0][1]), (bboxes[0][0][2], bboxes[0][0][3]), (0,255,0), 2)
        # cv2.imwrite('test.png', img)
        
        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        bboxes = torch.tensor(bboxes)
        masks = torch.tensor(masks).float()
        return image, bboxes, masks


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


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
