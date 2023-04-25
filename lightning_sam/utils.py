import os

import cv2
import torch
from box import Box
from dataset import COCODataset
from model import Model
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value."""

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


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def visualize(cfg: Box):
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
                          annotation_file=cfg.dataset.val.annotation_file,
                          transform=None)
    predictor = model.get_predictor()
    os.makedirs(cfg.out_dir, exist_ok=True)

    for image_id in tqdm(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
        cv2.imwrite(image_output_path, image_output)


if __name__ == "__main__":
    from config import cfg
    visualize(cfg)
