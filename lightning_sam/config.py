from box import Box

config = {
    "devices": [2],
    "batch_size": 2,
    "num_workers": 1,
    "num_epochs": 100,
    "eval_interval": 2,
    "log_n_steps": 10,
    "out_dir": "/home/mp/work/track_anything/segment_anything_tuning/out/checkpoints",
    "log_file": "/home/mp/work/track_anything/segment_anything_tuning/out/log_train.txt",
    "visualise_path": "/home/mp/work/track_anything/segment_anything_tuning/out/vis_val",
    "opt": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 2500,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "/home/au/segment-anything/weights/sam_vit_h_4b8939.pth",
        # "checkpoint": "/home/mp/work/track_anything/chpts/sam_vit_l_0b3195.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        # "train": {
        #     "img_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/images",
        #     "mask_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/masks/truck"
        # },
        # "val": {
        #     "img_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/images",
        #     "mask_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/masks/truck"
        # }
        "train": {
            "img_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/train/img",
            "mask_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/train/masks/truck"
        },
        "val": {
            "img_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/img",
            "mask_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/masks/truck"
        }
    }
}

cfg = Box(config)
