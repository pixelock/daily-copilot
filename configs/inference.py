# coding: utf-8
# @Author: pixelock
# @File: inference.py
# @Time: 2023/6/17 16:49

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    finetuning_type: Optional[str] = field(
        default='lora',
        metadata={
            'help': 'fine-tuning method',
            'choices': ['full', 'lora', 'ptv2'],
        }
    )
