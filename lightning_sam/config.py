import json
import argparse

from box import Box

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--config',
    dest='config',
    type=str,
    default='configs/coco2017.json',
    help='JSON config file'
)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = Box(json.load(f))
