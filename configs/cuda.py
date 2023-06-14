# coding: utf-8
# @File: cuda.py
# @Author: pixelock
# @Time: 2023/6/10 10:59

import torch

__all__ = ['DEVICE', 'NUM_GPU']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPU = torch.cuda.device_count() if DEVICE == 'cuda' else 0
