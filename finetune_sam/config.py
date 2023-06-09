from box import Box

config = {
    "devices": [0],
    "batch_size": 2,
    "num_workers": 1,
    "num_epochs": 50,
    "eval_interval": 2,
    "log_n_steps": 10,
    "out_dir": "/home/mp/work/track_anything/data/train_h/",
    "log_file": "log_train.txt",
    "opt": {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 2500,
    },
    "model": {
        # type and checkpoint must be of the same size of the model
        "type": 'vit_h',
        "checkpoint": "/home/au/segment-anything/weights/sam_vit_h_4b8939.pth",
        # "checkpoint": "/home/mp/work/track_anything/data/train_h/chpt/ckpt-ep-0020.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "generate_random_for_empty": True,
    "dataset": {
        "train": {
            "img_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/train/img",
            "mask_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/train/masks/truck"
        },
        "val": {
            "img_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/img",
            "mask_dir": "/home/mp/work/track_anything/data/dataset_trucks_segmentation/val/masks/truck"
        },
        "cache_path": "/home/mp/work/track_anything/.cache_dataset",
    }
}

cfg = Box(config)
