import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import cfg
import torchvision.transforms as transforms


class SAMMasks2BoxesDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None, generate_random_for_empty=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.file_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(self.img_dir)]
        self.cache_path = Path(cfg.dataset.cache_path)
        self.to_tensor = transforms.ToTensor()
        self.generate_random_for_empty = generate_random_for_empty

    def __len__(self):
        return len(self.file_names)
    
    @staticmethod
    def mask2masks_boxes(mask, generate_random_for_empty=False, max_num_random=2, scale_range=(0.1, 0.5)):
        # mask np.array (shape=[H,W])
        # generate_random_for_empty: generate_random box for empty mask

        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not len(mask_contours):
            if not generate_random_for_empty:
                bboxes = [[0, 0, mask.shape[1], mask.shape[0]]]
                masks = [np.zeros(mask.shape, dtype = "uint8")]
            else:
                num_now = np.random.randint(1, max(max_num_random, 1)+1)
                bboxes = []
                masks = [np.zeros(mask.shape, dtype = "uint8") for _ in range(num_now)]

                shape = np.array(mask.shape).astype(int)[::-1]
                
                for _ in range(num_now):
                    scale = np.random.uniform(*scale_range, size=2)
                    shape_scaled = (shape*scale).astype(int)
                    x0y0 = np.random.randint(0, shape-shape_scaled)
                    x1y1 = x0y0 + shape_scaled
                    bboxes.append(np.concatenate([x0y0, x1y1]))

        else:
            bboxes = []
            masks = []

            mask_contour_min_area = 100
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) < mask_contour_min_area:
                    continue
                bboxes.append(cv2.boundingRect(mask_contour))

                mask = np.zeros(mask.shape, dtype = "uint8")
                cv2.drawContours(mask, [mask_contour], 0, 1, -1)
                masks.append(mask)

            bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
        
        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0).astype(np.float32)

        return masks, bboxes

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, f'{file_name}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{file_name}.png')
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            if image is None:
                file_path = img_path
            else:
                file_path = mask_path
            raise RuntimeError(f'Cant open file: {file_path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks, bboxes = SAMMasks2BoxesDataset.mask2masks_boxes(mask, generate_random_for_empty=self.generate_random_for_empty)

        embedding = self.cache_path / f'{file_name}.npy'
        if embedding.is_file() or embedding.exists():
            embedding = np.load(embedding)
            embedding = torch.tensor(embedding)

        return image, bboxes, masks, embedding


def collate_fn(batch):
    images, bboxes, masks, embedings = zip(*batch)
    images = np.stack(images)
    bboxes = bboxes
    masks = masks
    return images, bboxes, masks, embedings



def load_datasets(cfg):
    
    train = SAMMasks2BoxesDataset(img_dir=cfg.dataset.train.img_dir,
                        mask_dir=cfg.dataset.train.mask_dir,
                        generate_random_for_empty=cfg.generate_random_for_empty)
    val = SAMMasks2BoxesDataset(img_dir=cfg.dataset.val.img_dir,
                      mask_dir=cfg.dataset.val.mask_dir,
                      generate_random_for_empty=cfg.generate_random_for_empty)
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
