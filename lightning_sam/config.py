from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 1,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "../out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_l',
        "checkpoint": "/home/mp/work/track_anything/chpts/sam_vit_l_0b3195.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
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
