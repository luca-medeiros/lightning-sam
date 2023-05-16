from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 1,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "/home/au/lightning-sam/out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/home/au/segment-anything/weights/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/images",
            "annotation_file": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/truck_ann_2.json"
        },
        "val": {
            "root_dir": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/images",
            "annotation_file": "/media/data3/au/tasks/2023_05_05_sam_labelling/finetune_data/truck_ann_2.json"
        }
    }
}

cfg = Box(config)
