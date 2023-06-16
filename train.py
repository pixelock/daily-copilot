# coding: utf-8
# @File: train.py
# @Author: pixelock
# @Time: 2023/6/13 21:16

import os
from transformers import set_seed
from transformers import HfArgumentParser
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

    set_seed(training_args.seed)

    if training_args.model_type == 'chatglm':
        from models.chatglm import ChatGLM
        model = ChatGLM.from_pretrained(training_args.model_name_or_path, config=model_args)




if __name__ == '__main__':
    train()
