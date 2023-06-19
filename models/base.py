# coding: utf-8
# @File: base.py
# @Author: pixelock
# @Time: 2023/6/10 11:00

from typing import Any, Optional
from abc import ABC, abstractmethod
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from configs.models import ModelConfig
from configs.training import FinetuningConfig
from configs.inference import InferenceConfig
from logger import logger


class BaseModel(ABC):
    model: Any = None
    tokenizer: Any = None

    def __init__(self,
                 model_args: Optional[ModelConfig] = None,
                 finetuning_args: Optional[FinetuningConfig] = None,
                 inference_args: Optional[InferenceConfig] = None):
        self.do_training = True if finetuning_args is not None else False
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.inference_args = inference_args

        self.prepare_env()
        self.tokenizer = self.prepare_tokenizer()
        if self.do_training:
            self.model = self.prepare_model_for_training()
        else:
            self.model = self.prepare_model_for_inference()
        self.load_adapter()

    @abstractmethod
    def prepare_env(self):
        pass

    @abstractmethod
    def prepare_tokenizer(self):
        pass

    @abstractmethod
    def prepare_model_for_training(self):
        pass

    @abstractmethod
    def prepare_model_for_inference(self):
        pass

    def load_adapter(self):
        if self.finetuning_args.finetuning_type == 'lora':
            self.model = self.load_adapter_lora(self.model)

    def load_adapter_lora(self, model):
        if self.do_training:
            if self.model_args.checkpoint_dir is not None:
                if self.model_args.resume_lora_training:
                    checkpoints_to_merge, last_checkpoint = \
                        self.model_args.checkpoint_dir[:-1], self.model_args.checkpoint_dir[-1]
                else:
                    checkpoints_to_merge, last_checkpoint = self.model_args.checkpoint_dir, None

                for checkpoint in checkpoints_to_merge:
                    model = PeftModel.from_pretrained(model, checkpoint)
                    model = model.merge_and_unload()
                    logger.info(f'merged lora model checkpoints from {checkpoint}')

                if last_checkpoint is not None:
                    model = PeftModel.from_pretrained(model, last_checkpoint, is_trainable=True)
                    logger.info(f'reload lora model checkpoints from {last_checkpoint}, and continue training')

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.finetuning_args.lora_rank,
                lora_alpha=self.finetuning_args.lora_alpha,
                lora_dropout=self.finetuning_args.lora_dropout,
                target_modules=self.finetuning_args.lora_target,
            )
            model = get_peft_model(model, lora_config)
        else:
            pass

        return model

    @classmethod
    def from_pretrained(cls,
                        model_args: Optional[ModelConfig] = None,
                        finetuning_args: Optional[FinetuningConfig] = None,
                        inference_args: Optional[InferenceConfig] = None):
        return cls(model_args, finetuning_args, inference_args)
