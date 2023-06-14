# coding: utf-8
# @File: models.py
# @Author: pixelock
# @Time: 2023/6/13 20:32

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: Optional[str] = None


@dataclass
class ChatGLMConfig(ModelConfig):
    multi_gpu: bool = False
    use_quant: bool = False
    quant_bit: str = 'int8'
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 2048
    pre_seq_len: Optional[int] = None
