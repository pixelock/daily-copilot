# coding: utf-8
# @File: cuda.py
# @Author: pixelock
# @Time: 2023/6/10 10:58

import torch
import pynvml
from pydantic import BaseModel
from typing import Dict, Union

from configs.cuda import NUM_GPU


class GPUStatus(BaseModel):
    index: int
    gpu_type: str
    total_mem: float
    used_mem: float
    free_mem: float


def torch_gc(device='cuda'):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def check_gpu_status(devices=None, verbose: bool = True):
    if not devices:
        devices = list(range(NUM_GPU))
    elif isinstance(devices, str):
        devices = [int(i) for i in devices.split(',')]
    elif isinstance(devices, int):
        devices = [devices]
    else:
        assert isinstance(devices, list), f'`devices` must be one of (int, str, list or None), got {devices} instead.'

    pynvml.nvmlInit()
    device_status = dict()
    for gpu_index in devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_type = pynvml.nvmlDeviceGetName(handle)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = men_info.total / 1024 ** 2
        used_mem = men_info.used / 1024 ** 2
        free_mem = men_info.free / 1024 ** 2

        device_status[int(gpu_index)] = GPUStatus(
            index=int(gpu_index),
            gpu_type=gpu_type,
            total_mem=total_mem,
            used_mem=used_mem,
            free_mem=free_mem,
        )

        if verbose:
            print(f'GPU {gpu_index} | Type: {gpu_type} | '
                  f'Total mem: {total_mem:.2f}MB | Used mem: {used_mem:.2f}MB | Free mem: {free_mem:.2f}MB')

    pynvml.nvmlShutdown()

    return device_status


def fetch_available_gpus(threshold: Union[int, float], devices=None, verbose: bool = True):
    gpu_status = check_gpu_status(devices=devices, verbose=verbose)

    available = sorted(
        [(k, v.free_mem) for k, v in gpu_status.items() if v.free_mem >= threshold],
        key=lambda x: x[1],
        reverse=True,
    )
    available_gpus = [x for x, y in available]
    return available_gpus
