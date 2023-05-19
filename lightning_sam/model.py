import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import numpy as np
import cv2
from  torchvision.transforms import Resize
import torch


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

    def forward(self, images, bboxes, embedings):
        _, _, H, W = images.shape
        idxs = []
        images_do_emb = []
        for i, (image, embeding) in enumerate(zip(images, embedings)):
            if not isinstance(embeding, torch.Tensor):
                idxs.append(i)
                images_do_emb.append(image)

        if images_do_emb:
            images_do_emb = torch.stack(images_do_emb)

            with torch.no_grad():
                done_embeddings = self.model.image_encoder(images_do_emb)
            torch.cuda.empty_cache()
            embedings = list(embedings)

            for idx, embeding in zip(idxs, done_embeddings):
                embeding_cache = embeding.detach().cpu().numpy()
                embeding_cache = np.ascontiguousarray(embeding_cache)
                np.save(embedings[idx], embeding_cache)
                embedings[idx] = embeding

        pred_masks = []
        ious = []
        for i, (embedding, box) in enumerate(zip(embedings, bboxes)):
            
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
