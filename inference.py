import torch
import cv2
import supervision as sv
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor

def generate_sam_masks(
    images_dir: str,
    predicted_masks_dir: str,
    output_dir: str,
    model_path: str,
    model_type: str = 'vit_h',
    device_id: int = 0,
    mask_extension: str = 'png',
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)    
    predicted_masks_dir = Path(predicted_masks_dir)
    
    device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else torch.device('cpu')
    sam = sam_model_registry[model_type](checkpoint=model_path).to(device)
    mask_predictor = SamPredictor(sam)
    
    masks_classes = [path.name for path in predicted_masks_dir.iterdir()]
    
    for image_path in tqdm(images_dir.iterdir()):
        image_rgb = cv2.imread(str(image_path), cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(image_rgb)
        
        for mask_class in masks_classes:
            
            result_masks_dir = output_dir / mask_class
            result_masks_dir.mkdir(exist_ok=True, parents=True)
            mask_name = f"{image_path.stem}.{mask_extension}"
            
            mask = cv2.imread(str(predicted_masks_dir / mask_class / mask_name), cv2.IMREAD_GRAYSCALE)
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_boxes = [cv2.boundingRect(cnt) for cnt in mask_contours]
            mask_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]) for box in mask_boxes]
            
            detected_masks = np.zeros(mask.shape, dtype=np.uint8)
            if len(mask_boxes) != 0:
                
                for mask_index in range(len(mask_boxes)):
                    detected_mask, _, _ = mask_predictor.predict(
                        box=mask_boxes[mask_index],
                        multimask_output=False
                    )
                    detected_mask = detected_mask.astype(np.uint8)[0]
                    detected_masks += detected_mask
            
            detected_masks[detected_masks >= 1] = 1
            cv2.imwrite(str(result_masks_dir / mask_name), detected_masks * 255)


if __name__ == "__main__":
    generate_sam_masks(
        model_type="vit_h",
        # model_path="/home/mp/work/track_anything/segment_anything_tuning/out/checkpoints/epoch-000009-f10.98-ckpt.pth",
        # model_path="/media/data3/au/tasks/2023_05_05_sam_labelling/out/epoch-000027-f10.85-ckpt.pth",
        # model_path="/home/mp/work/track_anything/segment_anything_tuning/out/checkpoints/epoch-000012-f10.98-ckpt.pth",
        model_path="/home/au/segment-anything/weights/sam_vit_h_4b8939.pth",
        images_dir="/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/img",
        predicted_masks_dir="/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/masks",
        output_dir="/media/data3/au/tasks/2023_05_22_count_sam_metrics/sam_preds_default",
        device_id=0,
    )