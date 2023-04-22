from box import Box

config = {
    "num_devices": 4,
    "batch_size": 8,
    "num_workers": 4,
    "num_epochs": 300,
    "eval_interval": 300,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 0,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/coco/coco2017/train2017",
            "annotation_file": "/coco/coco2017/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "/coco/coco2017/val2017",
            "annotation_file": "/coco/coco2017/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)