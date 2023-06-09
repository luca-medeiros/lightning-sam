import numpy as np
import cv2
import os
import shutil
from time import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
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
from model import SAM_finetuner
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
from typing import Tuple

import neptune

torch.set_float32_matmul_precision('high')

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
    
    @staticmethod
    def num_dig(num):
        return np.ceil(np.log10(num)).astype(int)
            

def configure_opt(cfg: Box, model: SAM_finetuner):
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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.int64, copy=True)
    color = [255,0,0]
    img[mask>0.5] += np.array(color)
    img[mask>0.5] //= 2
    img = img.astype(np.uint8)
    return img


def validate(fabric: L.Fabric, model: SAM_finetuner, val_dataloader: DataLoader, epoch: int = 0, logger=None, save_checkpoint=False) -> Tuple[float, float]:
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
    time_begin = time()
    vis_cntr = 0
    start = time()
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):

            images, bboxes, gt_masks, embedings = data
            num_images = images.shape[0]
            gt_masks = [torch.FloatTensor(gt_mask).to(fabric.device) for gt_mask in gt_masks]
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
            if iter > 0 and iter % cfg.log_n_steps == 0:
                val_log(logger,
                        f'Val: [{epoch}] - [{iter:>{Logger.num_dig(size)}}/{size}]: '
                        f'Time [{time() - start:2.3f}s]; '
                        f'Mean IoU: [{ious.avg:.4f}]; '
                        f'Mean F1: [{f1_scores.avg:.4f}]'
                )
                start = time()
            visualise_path = Path(cfg.out_dir) / 'vis_epochs' / f'ep_{epoch}'
            visualise_path.mkdir(parents=True, exist_ok=True)

            for img, pred_masks, bbox in zip(images, pred_masks, bboxes):
                                    
                # img = np.ascontiguousarray(img)
                pred_masks = np.ascontiguousarray(pred_masks[0].detach().cpu().numpy()) 
                # bbox = np.ascontiguousarray(bbox.detach().cpu().numpy()).astype(int)
                img = plot_mask_on_img(img, pred_masks)
                
                for bb in bbox:
                    img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                img_path = str((visualise_path / f'{vis_cntr}.png').absolute())
                cv2.imwrite(img_path, img)
                vis_cntr += 1

    if save_checkpoint:
        chpt_path = Path(f'{cfg.out_dir}/chpt')
        chpt_path.mkdir(parents=True, exist_ok=True)
        chpt_path = str(chpt_path.absolute())
        val_log(logger, f"Saving checkpoint to {chpt_path}", once=True)
        state_dict = model.model.state_dict()
        if fabric.global_rank == 0:
            torch.save(state_dict, os.path.join(chpt_path, f"ckpt-ep-{epoch:04d}.pth"))
    
    full_time = time()-time_begin
    val_log(logger, 
        f'Validation [{epoch}]:'
        f'  Full time: {full_time}'
        f'; Mean IoU: [{ious.avg:.4f}'
        f'; Mean F1: [{f1_scores.avg:.4f}]\n'
        + '#'*120 +  '\n',
        once=True
    )

    model.train()
    
    return ious.avg, f1_scores.avg

# cntr = 0

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: SAM_finetuner,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    run = neptune.init_run(
        project="alex-uv2/segment-anything",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MGQ0NzkzZS1jNGFiLTRmYjctOTg3Yi03NmEwZWE4ZGIwNzYifQ==",
    )
    
    run['cfg'] = (vars(cfg))
    
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    logger = Logger(os.path.join(cfg.out_dir, cfg.log_file), fabric)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dataset.cache_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.num_epochs+1):
        logger.log_once(f'Start traing epoch #{epoch}')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time()

        for iter, data in enumerate(train_dataloader):
            # break
            data_time.update(time() - end)
            images, bboxes, gt_masks, embedings = data
            
            batch_size = images.shape[0]
            gt_masks = [torch.FloatTensor(gt_mask).to(fabric.device) for gt_mask in gt_masks]
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
            batch_time.update(time() - end)
            end = time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)
            
            if iter > 0 and iter % cfg.log_n_steps == 0:
                size = len(train_dataloader)
                logger.log(f'Epoch: [{epoch}][{iter+1:>{Logger.num_dig(size)}}/{size}]'\
                        f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'\
                        f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'\
                        f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'\
                        f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'\
                        f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'\
                        f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'\
                        f' | LR [{scheduler.get_last_lr()[0]:.6f}]')

            
        run['train_total_loss'].log(total_losses.avg)
        run['train_dice_loss'].log(dice_losses.avg)
        run['train_focal_loss'].log(focal_losses.avg)
        run['train_iou_loss'].log(iou_losses.avg)
        run['train_lr'].log(scheduler.get_last_lr()[0])
            
        if epoch > 0 and epoch % cfg.eval_interval == 0:
            val_epoch_iou, val_epoch_f1 = validate(fabric, model, val_dataloader, epoch, logger=logger, save_checkpoint=True)
            
            run['val_iou'].log(val_epoch_iou)
            run['val_f1'].log(val_epoch_f1)
    run.stop()

def configure_opt(cfg: Box, model: SAM_finetuner):

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
        model = SAM_finetuner(cfg)

    train_data, val_data = load_datasets(cfg)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
