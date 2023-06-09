import os
import json
from box import Box
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np


from finetune_sam.config import cfg
from finetune_sam.model import SAM_finetuner
from finetune_sam.dataset import SAMMasks2BoxesDataset

from eg_data_tools.eg_data_tools.model_evaluation.segmentation_metrics import count_segmentation_metrics


def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


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

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        horizontal_lines.append((y, y + h))
    return horizontal_lines


@torch.no_grad()
def generate_sam_masks(
    cfg: Box,
    pred_masks_dir: str,
    vis_dir: str,
):
    images_dir: str = cfg.dataset.val.img_dir
    gt_masks_dir: str = cfg.dataset.val.mask_dir
    device_id: int = cfg.devices[0]
    cache_path: str = cfg.dataset.cache_path

    images_dir = Path(images_dir)
    pred_masks_dir = Path(pred_masks_dir)
    vis_dir = Path(vis_dir)
    cache_path = Path(cache_path)
    gt_masks_dir = Path(gt_masks_dir).parent
    
    device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else torch.device('cpu')
    sam = SAM_finetuner(cfg)
    sam.to(device)
    
    masks_classes = [path.name for path in gt_masks_dir.iterdir()]
    for mask_class in masks_classes:
        result_masks_dir = pred_masks_dir / mask_class
        result_masks_dir.mkdir(exist_ok=True, parents=True)
        result_vis_dir = vis_dir / mask_class
        result_vis_dir.mkdir(exist_ok=True, parents=True)

    for image_path in tqdm(list(images_dir.iterdir())):
        image = cv2.imread(str(image_path))
        assert image is not None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_rgb[None]
        embeding = cache_path / f'{image_path.stem}.npy'
        if embeding.is_file():
            embeding = np.load(embeding)
            embeding = [torch.tensor(embeding).to(device)]
        else:
            embeding = [embeding]
        
        for mask_class in masks_classes:

            result_masks_dir = pred_masks_dir / mask_class
            result_vis_dir = vis_dir / mask_class
            
            mask_name = f"{image_path.stem}.png"
            gt_mask = cv2.imread(str(gt_masks_dir / mask_class / mask_name), cv2.IMREAD_GRAYSCALE)
            assert gt_mask is not None, mask_name
            _, bboxes = SAMMasks2BoxesDataset.mask2masks_boxes(
                gt_mask, generate_random_for_empty=cfg.generate_random_for_empty)

            detected_masks, _ = sam(image, bboxes, embeding)
            detected_mask = (detected_masks[0]).detach().cpu().numpy()
            detected_mask = np.bitwise_and.reduce(detected_mask>0.5, axis=0).astype(np.uint8)
            # detected_mask = np.ascontiguousarray(detected_mask)
            cv2.imwrite(str(result_masks_dir / mask_name), detected_mask)

            img_write = plot_mask_on_img(image_rgb, detected_mask)
            for bb in bboxes:
                img_write = cv2.rectangle(img_write, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            save_path = str(result_vis_dir / mask_name)
            cv2.imwrite(save_path, img_write)


if __name__ == "__main__":
    set_seed(seed=1337)
    out_dir = '/home/mp/work/track_anything/data/infer_base/'

    out_dir = Path(out_dir)
    pred_masks_dir = str(out_dir / 'masks')
    vis_dir = str(out_dir / 'vis')

    os.system(f'rm -rf {out_dir/"*"}')
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(pred_masks_dir).mkdir(exist_ok=True)
    Path(vis_dir).mkdir(exist_ok=True)

    with torch.inference_mode():
        generate_sam_masks(
            cfg=cfg,
            pred_masks_dir=pred_masks_dir,
            vis_dir=vis_dir,
        )

    images_dir: str = cfg.dataset.val.img_dir
    gt_masks_dir: str = cfg.dataset.val.mask_dir
    gt_masks_dir = str(Path(gt_masks_dir).parent.absolute())

    result_metrics = count_segmentation_metrics(
        img_dir=images_dir,
        gt_masks_dir=gt_masks_dir,
        pred_masks_dir=pred_masks_dir,
    )
    with open(str(out_dir/'res.json'), 'w') as f:
        json.dump(result_metrics, f)
