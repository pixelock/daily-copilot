# coding: utf-8
# @File: base.py
# @Author: pixelock
# @Time: 2023/6/10 11:00

from typing import Any, Optional
from abc import ABC, abstractmethod

from configs.models import ModelConfig
from configs.training import FinetuningConfig
from configs.inference import InferenceConfig


class BaseModel(ABC):
    model: Any = None
    tokenizer: Any = None

    def __init__(self,
                 model_args: Optional[ModelConfig] = ModelConfig,
                 finetuning_args: Optional[ModelConfig] = FinetuningConfig,
                 inference_args: Optional[InferenceConfig] = InferenceConfig):
        self.do_training = True if finetuning_args is not None else False
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.inference_args = inference_args

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def load_original_model(self):
        pass



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
