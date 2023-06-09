# Finetune SAM


The main task for this finetuning is to finetune the SAM model to segment your custom data based on promt bounding boxes. Further, this tool can autogenerate segmentation masks on unlabeled data via the following pipeline.

![pipeline](imgs/pipeline.png)

## Installation

```
git clone https://github.com/everguard-inc/segment_anything_tuning
cd segment_anything_tuning
conda create -n "fineSAM" python=3.9 -y
caonda activate fineSAM
pip install -r requirements.txt

```

## Quick Start

1. Prepare your custom dataset. The recommended structure should be the following
    ```
    ├── train
    │   ├── img
    │   │   ├── train_img1.jpeg
    │   │   ...
    │   └── masks
    │   │   ├── train_mask1.png
    │   │   ...
    ├── val
    │   ├── img
    │   │   ├── val_img1.jpeg
    │   │   ...
    │   └── masks
    │   │   ├── val_mask1.png
    │   │   ...
    ```
    Or any other, you anyway need to set a path to every of four dirs in `finetune_sam/config.py`.

2. Edit `finetune_sam/config.py` with your dataset paths. Also, check other params:cuda-devices-ID, out-path, etc.

3. Run ```finetune_sam/train.py```

4. To inference eg_dataset (in folder `segment_anything_tuning`; `fineSAM` - activated):
```
pwd # returns ..../segment_anything_tuning
export PYTHONPATH="${PYTHONPATH}:$(pwd)/finetune_sam"
python inference.py
```

## Features
- Supports custom datasets
- Performed caching of embeddings:
    - If embedding is not cached: 1-2 sec/img
    - If embedding cached: 0.2-0.4 sec/img
- Image preprocess encapsulated into the model. As input into the model, you must pass raw RGB image and promt bounding boxes for each image, both in patched form.
- Setuped Neptune for tracking training progress.

## Train plots
- train_dice_loss
![train_dice_loss](imgs/train_dice_loss.png)
- train_focal_loss
![train_focal_loss](imgs/train_focal_loss.png)
- train_iou_loss
![train_iou_loss](imgs/train_iou_loss.png)
- val_f1
![val_f1](imgs/val_f1.png)
- val_iou
![val_iou](imgs/val_iou.png)



## Results
Trained on [dataset](https://github.com/everguard-inc/dataset_trucks_segmentation/tree/zekelman_ontario_tents) of tracks (~3500 imgs - train set, ~700 imgs - test set). After 20-th epoch metrics are mainly stable. Intresting is tha it dows not overfit after huge amoun of pochs (100).

| Class  |    IoU  | dice    | Epoch |
| ------ | ------- | ------- | ----- |
| crane  | 0.3520  | 0.2905  |  0    |
| tent   | 0.2966  | 0.2663  |  0    |
| truck  | 0.6643  | 0.7808  |  0    |
| net    | 0.4888  | 0.4572  |  0    |
| total  | 0.6000  | 0.5659  |  0    |
| ------ | ------- | ------- | ----- |
| crane  | 0.4848  | 0.4724  |  10   |
| tent   | 0.7539  | 0.7163  |  10   |
| truck  | 0.9412  | 0.9166  |  10   |
| net    | 0.2203  | 0.1581  |  10   |
| total  | 0.6000  | 0.5659  |  10   |
| ------ | ------- | ------- | ----- |
| crane  | 0.4978  | 0.4873  |  20   |
| tent   | 0.8469  | 0.8120  |  20   |
| truck  | 0.9586  | 0.9428  |  20   |
| net    | 0.1576  | 0.1181  |  20   |
| total  | 0.6152  | 0.5900  |  20   |


# TODO
- For generating bounding boxes from text and prompt them to SAM, you may check: [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
- Add noise to promt boxes for better adapting for low-quality promts.
- Add training on different classes simultaneously.


## Resources
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [Lightning Segment-Anything Model](https://github.com/luca-medeiros/lightning-sam)

## License
This project is licensed the same as the SAM model.

## Notes
- Uses the original implementation of SAM.
- Loss calculated as stated on the paper (20 * focal loss + dice loss + mse loss).
- Only supports bounding box input prompts.