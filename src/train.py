import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, global_iter: int):
    model.eval()
    ious = AverageMeter()
    precisions = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
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
                f'Val: {iter}/{len(val_dataloader)}: Mean IoU: {ious.avg:.4f} Mean Precision: {precisions.avg:.4f}, Mean F1: {f1_scores.avg:.4f}'
            )

    fabric.print(
        f'Validation: Mean IoU: {ious.avg:.4f} Mean Precision: {precisions.avg:.4f}, Mean F1: {f1_scores.avg:.4f}')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir,
                                            f"global_iter-{global_iter:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    global_iter = 0

    for epoch in range(cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()

        for iter, data in enumerate(train_dataloader):
            if global_iter > 0 and global_iter % cfg.eval_interval == 0:
                validate(fabric, model, val_dataloader, global_iter)

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = forward_pass(model, images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = fabric.to_device(torch.tensor(0.))
            loss_dice = fabric.to_device(torch.tensor(0.))
            loss_iou = fabric.to_device(torch.tensor(0.))
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            global_iter += 1

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch+1}][{iter+1}/{len(train_dataloader)}] '
                         f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
                         f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                         f'Focal Loss {focal_losses.val:.4f} ({focal_losses.avg:.4f})\t'
                         f'Dice Loss {dice_losses.val:.4f} ({dice_losses.avg:.4f})\t'
                         f'IoU Loss {iou_losses.val:.4f} ({iou_losses.avg:.4f})\t'
                         f'Total Loss {total_losses.val:.4f} ({total_losses.avg:.4f})\t')


def forward_pass(model, images, bboxes):
    _, _, H, W = images.shape
    image_embeddings = model.image_encoder(images)
    pred_masks = []
    ious = []
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
        ious.append(iou_predictions)
    # binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    return pred_masks, ious


def main(cfg) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint)
        model.train()
    if cfg.model.freeze.image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if cfg.model.freeze.prompt_encoder:
        for param in model.prompt_encoder.parameters():
            param.requires_grad = False
    if cfg.model.freeze.mask_decoder:
        for param in model.mask_decoder.parameters():
            param.requires_grad = False

    train_data, val_data = load_datasets(cfg, model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer = torch.optim.Adam(model.mask_decoder.parameters(),
                                 lr=cfg.opt.learning_rate,
                                 weight_decay=cfg.opt.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, train_data, val_data)


if __name__ == "__main__":
    main(cfg)
