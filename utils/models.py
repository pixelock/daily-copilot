# coding: utf-8
# @File: models.py
# @Author: pixelock
# @Time: 2023/6/25 23:50

import torch
from typing import Dict


def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:  # get state dict containing trainable parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict
