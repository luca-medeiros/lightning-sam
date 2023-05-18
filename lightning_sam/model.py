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
    img[mask>0] += np.array(color)
    img[mask>0] //= 2
    img = img.astype(np.uint8)
    return img

imgs_idx = 0

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

            img = images[i].clone().cpu().permute(1,2,0).numpy() * 255
            img = np.ascontiguousarray(img)

            box_draw = box.clone().cpu().numpy().astype(int)
            masks_draw = masks.clone().cpu().detach().numpy()
            masks_draw = np.ascontiguousarray(masks_draw)

            global imgs_idx

            img = plot_mask_on_img(img, masks_draw[0][0])
            img = cv2.rectangle(img, (box_draw[0][0], box_draw[0][1]), (box_draw[0][2], box_draw[0][3]), (0,255,0), 2)
            out_dir = '/home/mp/work/track_anything/segment_anything_tuning/lightning_sam/data/out/mask_pred'
            out_path = f'{out_dir}/{imgs_idx}.png'
            imgs_idx += 1
            cv2.imwrite(out_path, img)
        
        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
