import os
import torch
import time
import lightning as L
import torch.nn.functional as F

from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp

from losses import FocalLoss, DiceLoss
from dataset import load_datasets

out_dir = "out/training"

# Hyperparameters
learning_rate = 1e-4
weight_decay = 0
num_epochs = 300
eval_interval = 3
freeze_image_encoder = True
freeze_prompt_encoder = True
freeze_mask_decoder = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(fabric, model, val_dataloader):
    model.eval()
    ious = AverageMeter()
    precisions = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for data in val_dataloader:
            images, bboxes, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = forward_pass(model, images, bboxes)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_precision = smp.metrics.precision(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                precisions.update(batch_precision, num_images)
                f1_scores.update(batch_f1, num_images)

    fabric.print(
        f'Validation: Mean IoU: {ious.avg:.4f} Mean Precision: {precisions.avg:.4f}, Mean F1: {f1_scores.avg:.4f}')

    model.train()
    return f1_scores.avg


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
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        # evaluate the loss on train/val sets and write checkpoints
        if epoch > 0 and epoch % eval_interval == 0:
            mean_iou = validate(fabric, model, val_dataloader)

            fabric.print(f"epoch {epoch}: f1_scores {mean_iou:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            state_dict = model.state_dict()
            if fabric.global_rank == 0:
                torch.save(state_dict, os.path.join(out_dir, f"epoch-{epoch:06d}-ckpt.pth"))

        for iter, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data
            pred_masks, iou_predictions = forward_pass(model, images, bboxes)
            num_masks = sum([len(pred_mask) for pred_mask in pred_masks])
            focal_loss_ = torch.stack([focal_loss(pred, gt, num_masks) for pred, gt in zip(pred_masks, gt_masks)]).sum()
            dice_loss_ = torch.stack([dice_loss(pred, gt, num_masks) for pred, gt in zip(pred_masks, gt_masks)]).sum()
            total_loss_ = focal_loss_ + dice_loss_
            optimizer.zero_grad()
            fabric.backward(total_loss_)
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            focal_losses.update(focal_loss_.item(), images.size(0))
            dice_losses.update(dice_loss_.item(), images.size(0))
            total_losses.update(total_loss_.item(), images.size(0))

            fabric.print(f'Epoch: [{epoch}][{iter}/{len(train_dataloader)}] '
                         f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
                         f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                         f'Focal Loss {focal_losses.val:.4f} ({focal_losses.avg:.4f})\t'
                         f'Dice Loss {dice_losses.val:.4f} ({dice_losses.avg:.4f})\t'
                         f'Total Loss {total_losses.val:.4f} ({total_losses.avg:.4f})\t')


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


if __name__ == "__main__":
    main('vit_h', 'sam_vit_h_4b8939.pth')