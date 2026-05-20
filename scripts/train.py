"""Train LookingGPT2 on an image-caption dataset from HuggingFace.

Single-GPU:
    python scripts/train.py

Multi-GPU (all GPUs on one node):
    accelerate launch scripts/train.py

Quick smoke test (20 steps, no download needed):
    python scripts/train.py --max_steps 20 --batch_size 4 --mixed_precision no

Watch training:
    tensorboard --logdir runs/coco/tensorboard
"""

import argparse
import dataclasses

from talking_vit.trainers import TrainConfig, Trainer


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train LookingGPT2 on image-caption data")

    for f in dataclasses.fields(defaults):
        val = getattr(defaults, f.name)
        t = type(val)
        if t is bool:
            parser.add_argument(f"--{f.name}", default=val, action=argparse.BooleanOptionalAction)
        elif val is None:
            arg_type = str if "str" in str(f.type) else int
            parser.add_argument(f"--{f.name}", default=val, type=arg_type)
        else:
            parser.add_argument(f"--{f.name}", default=val, type=t)

    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    config = parse_args()
    Trainer(config).train()
