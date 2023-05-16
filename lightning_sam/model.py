import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import numpy as np
import cv2
from  torchvision.transforms import Resize
import torch

def plot_mask_on_img(img, mask):
    # img: h,w,c
    # masks: h,w
    img = img.astype(np.int64, copy=True)
    color = [255,0,0]
    img[mask>=1] += np.array(color)
    img[mask>0] //= 2
    img = img.astype(np.uint8)
    return img


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(images)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        pred_masks = []
        ious = []
        for i, (embedding, box) in enumerate(zip(image_embeddings, bboxes)):
            
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None,
                )
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            low_res_masks = low_res_masks.max(0)[0][None, ...]
            iou_predictions = iou_predictions.mean(0)[None, ...]
            
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
        
        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
