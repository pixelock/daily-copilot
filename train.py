# coding: utf-8
# @File: train.py
# @Author: pixelock
# @Time: 2023/6/13 21:16

import os
from transformers import set_seed
from transformers import HfArgumentParser, TrainingArguments
import torch

from configs.data import DataConfig
from configs.models import ModelConfig
from configs.training import FinetuningConfig


def train():
    t_parser = HfArgumentParser(ModelConfig)
    t_model_args, _ = t_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if t_model_args.model_type == 'chatglm':
        from configs.models import ChatGLMConfig
        T_ModelConfig = ChatGLMConfig
    else:
        raise ValueError(f'unknown model type: {t_model_args.model_type}')

    parser = HfArgumentParser([T_ModelConfig, DataConfig, FinetuningConfig, TrainingArguments])
    model_args, data_args, finetuning_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # global_rank = torch.distributed.get_rank()
    #
    # set_seed(training_args.seed)

    if training_args.model_type == 'chatglm':
        from models.chatglm import ChatGLM
        model = ChatGLM.from_pretrained(training_args.model_name_or_path, config=model_args)




if __name__ == '__main__':
    train()
