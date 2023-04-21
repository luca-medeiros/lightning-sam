import os
import torch

import lightning as L
import torch.nn.functional as F

from typing import Callable, Dict
from lightning.fabric.fabric import _FabricOptimizer

from segment_anything import sam_model_registry
from torch.utils.data import DataLoader

from losses import FocalLoss, DiceLoss
from dataset import load_datasets

out_dir = "out/training"

# Hyperparameters
learning_rate = 1e-4
weight_decay = 0
num_epochs = 100
eval_interval = 3
freeze_image_encoder = True
freeze_prompt_encoder = True
freeze_mask_decoder = False


def main(model_type, checkpoint) -> None:
    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed", strategy="ddp")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    with fabric.device:
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.train()
    if freeze_image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if freeze_prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False
    if freeze_mask_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    losses = {'focal_loss': focal_loss, 'dice_loss': dice_loss}

    train_sam(fabric, model, optimizer, losses, train_data, val_data)


def train_sam(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: _FabricOptimizer,
    losses: Dict[str, Callable],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    """The SAM training loop."""
    device = fabric.device

    for epoch in range(num_epochs):
        epoch_losses = []

        # evaluate the loss on train/val sets and write checkpoints
        if epoch > 0 and epoch % eval_interval == 0:
            # val_loss = validate(fabric, model, val_dataloader)
            val_loss = 0
            fabric.print(f"epoch {epoch}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            state_dict = model.state_dict()
            if fabric.global_rank == 0:
                torch.save(state_dict, os.path.join(out_dir, f"iter-{epoch:06d}-ckpt.pth"))

        for data in train_dataloader:
            binary_mask, gt_binary_mask = forward_pass(data, model, device)
            total_loss = torch.tensor([0.0])
            for loss_name, loss_fn in losses.items():
                loss = loss_fn(binary_mask, gt_binary_mask)
                fabric.log(loss_name, loss.item())
                total_loss += loss
            optimizer.zero_grad()
            fabric.backward(total_loss)
            optimizer.step()
            fabric.log("train_loss", total_loss.item())
            # epoch_losses.append(total_loss.item())


def forward_pass(data, model, device):
    images, bboxes, gt_masks = data
    _, _, H, W = images.shape
    image_embeddings = model.image_encoder(images)
    pred_masks = []
    for embedding, bbox in zip(image_embeddings, bboxes):
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks,
            (H, W),
            mode="bilinear",
            align_corners=False,
        )
        pred_masks.append(masks)
    # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    return pred_masks, gt_masks