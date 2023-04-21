import os
import torch
import lightning as L
import torch.nn.functional as F

from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger

from losses import FocalLoss, DiceLoss
from dataset import load_datasets

out_dir = "out/training"

# Hyperparameters
learning_rate = 1e-4
weight_decay = 0
num_epochs = 100
eval_interval = 10
freeze_image_encoder = True
freeze_prompt_encoder = True
freeze_mask_decoder = False


def main(model_type, checkpoint) -> None:
    fabric = L.Fabric(accelerator="cuda", devices=2, strategy="ddp", loggers=TensorBoardLogger(out_dir))
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

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
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(fabric, model, optimizer, train_data, val_data)


def train_sam(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    for epoch in range(num_epochs):
        epoch_losses = {'total_loss': [], 'dice_loss': [], 'focal_loss': []}

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
            images, bboxes, gt_masks = data
            pred_masks, iou_predictions = forward_pass(model, images, bboxes)
            num_masks = sum([len(pred_mask) for pred_mask in pred_masks])
            focal_loss_ = torch.stack([focal_loss(pred, gt, num_masks) for pred, gt in zip(pred_masks, gt_masks)]).sum()
            dice_loss_ = torch.stack([dice_loss(pred, gt, num_masks) for pred, gt in zip(pred_masks, gt_masks)]).sum()
            total_loss_ = focal_loss_ + dice_loss_
            fabric.log('focal_loss', focal_loss_.item())
            fabric.log('dice_loss', dice_loss_.item())
            fabric.log("train_loss", total_loss_.item())
            optimizer.zero_grad()
            fabric.backward(total_loss_)
            optimizer.step()

            epoch_losses['total_loss'].append(total_loss_.item())
            epoch_losses['dice_loss'].append(dice_loss_.item())
            epoch_losses['focal_loss'].append(focal_loss_.item())
        # fabric.print(f'Epoch {epoch}: total_loss: {epoch_losses['total_loss'].mean()}')


def forward_pass(model, images, bboxes):
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
            image_embeddings=embedding.unsqueeze(0),
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
        pred_masks.append(masks.squeeze(1))
    # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    return pred_masks, iou_predictions


if __name__ == "__main__":
    main('vit_h', 'sam_vit_h_4b8939.pth')