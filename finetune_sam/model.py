import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, SamPredictor

class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, images, bboxess, device):
        # Resize image and masks
        og_h, og_w = images.shape[-3:-1]
        
        images = [
            self.to_tensor(self.transform.apply_image(image))
            for image in images]
        bboxess =  [
            self.transform.apply_boxes(bboxes, (og_h, og_w))
            for bboxes in bboxess]

        # Pad image and masks to form a square
        h, w = images[0].size()[-2:]
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        transform_pad = transforms.Pad(padding)
        images = [transform_pad(image).to(device) for image in images]
        bboxess = [
            torch.FloatTensor(np.array([[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
            for bbox in bboxes])).to(device)
            for bboxes in bboxess
        ]

        return images, bboxess


class SAM_finetuner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print('Start loading SAM finetuner ... ', end='', flush=True)
        self.model_type = cfg.model.type
        self.model_path = cfg.model.checkpoint
        self.freeze_image_encoder = cfg.model.freeze.image_encoder
        self.freeze_prompt_encoder = cfg.model.freeze.prompt_encoder
        self.freeze_mask_decoder = cfg.model.freeze.mask_decoder
        self._setup()
        self.transform = ResizeAndPad(self.model.image_encoder.img_size)
        print('done')
        
    def _setup(self):
        self.model = sam_model_registry[self.model_type](checkpoint=self.model_path)
        self.model.train()
        if self.freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
    
    def _process_embedings(self, images, embedings):
        if embedings is not None:
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

                embedings = list(embedings)

                for idx, embeding in zip(idxs, done_embeddings):
                    embeding_cache = embeding.detach().cpu().numpy()
                    embeding_cache = np.ascontiguousarray(embeding_cache)
                    np.save(embedings[idx], embeding_cache)
                    embedings[idx] = embeding
        else:
            with torch.no_grad():
                embedings = self.model.image_encoder(images)
        return embedings
    
    def _preprocess(self, image:torch.tensor, bboxes:torch.tensor):
        image, bboxes = self.transform(image, bboxes, device=self.model.device)
        return image, bboxes
    
    def _postprocess(self, low_res_masks, size):
        # size=(H, W)
        max_size = max(size)

        pad_h = (max_size - size[0]) // 2
        pad_w = (max_size - size[1]) // 2
        
        
        masks = F.interpolate(
            low_res_masks,
            (max_size, max_size),
            mode="bilinear",
            align_corners=True,
        )
        masks = masks.squeeze(1)
        masks = masks[..., pad_h:max_size-pad_h, pad_w:max_size-pad_w]
        return masks

    def forward(self, images, bboxess, embedings=None):
        global cnt
        H, W = images.shape[1:3]

        images, bboxess = self._preprocess(images, bboxess)

        embedings = self._process_embedings(images, embedings)

        pred_masks = []
        ious = []
        for i, (embedding, bboxes) in enumerate(zip(embedings, bboxess)):
            
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=bboxes,
                    masks=None,
                )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = self._postprocess(low_res_masks, size=(H,W))
            pred_masks.append(masks)
            ious.append(iou_predictions)
        
        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
