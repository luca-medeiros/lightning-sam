# Lightning Segment-Anything Model

This library allows you to fine-tune the powerful Segment-Anything model from MetaAI for your custom COCO-format datasets. The library is built on top of Lightning AI's Fabric framework, providing an efficient and easy-to-use implementation for achieving state-of-the-art instance segmentation results.

## Features

- Supports custom COCO-format datasets
- Built on Lightning AI's Fabric framework
- Efficient fine-tuning of Segment-Anything model from MetaAI
- Includes training and validation loops

## Installation

```
git clone https://github.com/luca-medeiros/lightning-sam.git
cd lightning-sam
pip install .
```

## Quick Start

1. Prepare your custom COCO-format dataset. The dataset should include a JSON file with annotations and an images folder with corresponding images.

1. Edit src/config.py with your dataset paths.

1. Run src/train.py.

## Notes

- Only supports bounding box input prompts.

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)

## License

This project is licensed same as SAM model.
