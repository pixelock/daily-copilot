# coding: utf-8
# @File: base.py
# @Author: pixelock
# @Time: 2023/6/10 11:00

from typing import Any
from abc import ABC, abstractmethod


class BaseModel(ABC):
    model: Any = None
    tokenizer: Any = None

    @abstractmethod
    def init_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_tokenizer(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        pass
