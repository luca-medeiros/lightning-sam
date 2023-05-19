import numpy as np
import cv2
import os
import shutil
import time
from datetime import datetime
from tqdm import tqdm

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
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')

def num_dig(num):
    return np.ceil(np.log10(num)).astype(int)

class Logger:
    def __init__(self, log_file, fabric):
        self.log_file = log_file
        self.fabric = fabric
        self.log_onece = fabric.global_rank==0

    def log(self, msg):
        self.fabric.print(msg)
        time_now = datetime.now().strftime("[%d/%m/%Y %H:%M:%S]")
        msg = f'{time_now} {msg}'
        with open(self.log_file, 'a') as f:
            f.write(msg+'\n')


    def log_once(self, msg):
        if self.log_onece:
            self.log(msg)
            

def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

def plot_mask_on_img(img, mask):
    # img: h,w,c
    # masks: h,w
    img = img.astype(np.int64, copy=True)
    color = [255,0,0]
    img[mask>=1] += np.array(color)
    img[mask>0] //= 2
    img = img.astype(np.uint8)
    return img


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0, logger=None, save_checkpoint=False, visualise_path=''):
    def val_log(logger, msg, once=False):
        if isinstance(logger, Logger):
            if once:
                logger.log_once(msg)
            else:
                logger.log(msg)
        elif not once or fabric.global_rank==0:
            print(msg)

    
    val_log(logger, f'Start validation after epoch #{epoch}', once = True)

    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    if visualise_path:
        val_log(logger, f'visualazing in folder: {visualise_path}')
        if fabric.global_rank == 0:
            shutil.rmtree(visualise_path, ignore_errors=True)
            os.makedirs(visualise_path)
        cntr = 0
    
    time_begin = time.time()
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):

            images, bboxes, gt_masks, embedings = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes, embedings)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            size = len(val_dataloader)
            if iter % cfg.log_n_steps == 0:
                val_log(logger,
                        f'Val: [{epoch}] - [{iter:>{num_dig(size)}}/{size}]: '
                        f'Mean IoU: [{ious.avg:.4f}]; '
                        f'Mean F1: [{f1_scores.avg:.4f}]'
            )
            if visualise_path:
                for img, pred_masks, bbox in zip(images, pred_masks, bboxes):
                    img = np.ascontiguousarray((img.detach()*255).permute(1,2,0).cpu().numpy())
                    pred_masks = np.ascontiguousarray(pred_masks[0].detach().cpu().numpy()) 
                    bbox = np.ascontiguousarray(bbox.detach().cpu().numpy()).astype(int)
                    img = plot_mask_on_img(img, pred_masks)
                    
                    for bb in bbox:
                        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                    img_path = os.path.join(visualise_path, f'{fabric.global_rank}_{cntr}.png')
                    cv2.imwrite(img_path, img)
                    cntr += 1

    full_time = time.time()-time_begin
    val_log(logger, 
        f'Validation [{epoch}]:'
        f'  Full time: {full_time}'
        f'; Mean IoU: [{ious.avg:.4f}'
        f'; Mean F1: [{f1_scores.avg:.4f}]\n'
        + '#'*120 +  '\n',
        once=True
    )

    if save_checkpoint:
        val_log(logger, f"Saving checkpoint to {cfg.out_dir}", once=True)
        state_dict = model.model.state_dict()
        if fabric.global_rank == 0:
            torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    logger = Logger(cfg.log_file, fabric)

    for epoch in range(1, cfg.num_epochs):
        logger.log_once(f'Start traing epoch #{epoch}')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        bach_time_start = time.time()


        for iter, data in enumerate(train_dataloader):
            torch.cuda.empty_cache()

            data_time.update(time.time() - end)
            images, bboxes, gt_masks, embedings = data
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes, embedings)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            size = len(train_dataloader)
            if iter % cfg.log_n_steps == 0:
                logger.log(f'Epoch: [{epoch}][{iter+1:>{num_dig(size)}}/{size}]'\
                        f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'\
                        f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'\
                        f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'\
                        f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'\
                        f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'\
                        f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'\
                        f' | LR [{scheduler.get_last_lr()[0]:.6f}]')

        epoch_time = time.time() - bach_time_start
        logger.log_once(f'\nTrain end epoch: [{epoch}]'\
                        f' | Full time: [{epoch_time:.3f}s]'
                        f' | Time [({batch_time.avg:.3f}s]'\
                        f' | Data [({data_time.avg:.3f}s]'\
                        f' | Focal Loss [{focal_losses.avg:.4f}]'\
                        f' | Dice Loss [{dice_losses.avg:.4f}]'\
                        f' | IoU Loss [{iou_losses.avg:.4f}]'\
                        f' | Total Loss [{total_losses.avg:.4f}]\n'
                        + '#'*120 + '\n')
        if epoch > 0 and epoch % cfg.eval_interval == 0:
            vis_path = os.path.join(cfg.visualise_path, f'epoch_{epoch}')
            validate(fabric, model, val_dataloader, epoch, visualise_path=vis_path, logger=logger, save_checkpoint=False)

def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="cuda",
                      devices=cfg.devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
