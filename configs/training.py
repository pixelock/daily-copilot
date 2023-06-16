# coding: utf-8
# @File: training.py
# @Author: pixelock
# @Time: 2023/6/13 22:35

import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class TrainingConfig(TrainingArguments):
    model_type: str = 'chatglm'
    use_lora: bool = True
    use_quant: bool = True
    quant_bit: Optional[str] = 'int8'
    last_checkpoint: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.output_dir and os.path.isdir(self.output_dir) and self.do_train and not self.overwrite_output_dir:
            self.last_checkpoint = get_last_checkpoint(self.output_dir)
            if self.last_checkpoint is not None and len(os.listdir(self.output_dir)):
                raise ValueError(
                    f"Output directory ({self.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif self.last_checkpoint is not None and self.resume_from_checkpoint is None:
                print(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
