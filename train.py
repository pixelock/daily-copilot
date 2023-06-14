# coding: utf-8
# @File: train.py
# @Author: pixelock
# @Time: 2023/6/13 21:16

import os
from transformers import set_seed
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
import torch

from configs.training import TrainingConfig
from configs.data import DataConfig


def train():
    t_parser = HfArgumentParser(TrainingConfig)
    t_training_args, _ = t_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if t_training_args.model_type == 'chatglm':
        from configs.models import ChatGLMConfig
        ModelConfig = ChatGLMConfig
    else:
        raise ValueError(f'unknown model type: {t_training_args.model_type}')

    lora_args = None
    if t_training_args.use_lora:
        from peft import LoraConfig
        parser = HfArgumentParser([ModelConfig, DataConfig, TrainingConfig, LoraConfig])
        model_args, data_args, training_args, lora_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True,
        )
    else:
        parser = HfArgumentParser([ModelConfig, DataConfig, TrainingConfig])
        model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    global_rank = torch.distributed.get_rank()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    # int8 is not compatible with DeepSpeed (require not to pass device_map)
    if training_args.use_int8_training:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"

    if training_args.model_type == 'chatglm':
        from models.chatglm import ChatGLM
        model = ChatGLM.from_pretrained(training_args.model_name_or_path, config=model_args)




if __name__ == '__main__':
    train()
