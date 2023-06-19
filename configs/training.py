# coding: utf-8
# @File: training.py
# @Author: pixelock
# @Time: 2023/6/13 22:35

import os
import json
from typing import Optional, Union, List
from dataclasses import dataclass, field, asdict
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class FinetuningConfig:
    finetuning_type: Optional[str] = field(
        default='lora',
        metadata={
            'help': 'fine-tuning method',
            'choices': ['full', 'lora', 'ptv2'],
        }
    )
    use_gradient_checkpointing: bool = True

    """Lora fine-tuning arguments"""
    lora_rank: int = field(default=8, metadata={'help': 'lora attention dimension'})
    lora_alpha: int = field(default=None, metadata={'help': 'lora alpha'})
    lora_dropout: float = field(default=None, metadata={'help': 'lora dropout'})
    lora_target: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            'help': "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    """P-tuning fine-tuning arguments"""
    pre_seq_len: Optional[int] = field(
        default=16,
        metadata={"help": "Number of prefix tokens to use for P-tuning V2."}
    )
    prefix_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in P-tuning V2 or not."}
    )

    def save_to_json(self, json_path: str):
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return cls(**json.loads(text))


# @dataclass
# class TrainingConfig(TrainingArguments):
#     model_type: str = 'chatglm'
#     use_lora: bool = True
#     use_quant: bool = True
#     quant_bit: Optional[str] = 'int8'
#     last_checkpoint: Optional[str] = None
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         if self.output_dir and os.path.isdir(self.output_dir) and self.do_train and not self.overwrite_output_dir:
#             self.last_checkpoint = get_last_checkpoint(self.output_dir)
#             if self.last_checkpoint is not None and len(os.listdir(self.output_dir)):
#                 raise ValueError(
#                     f"Output directory ({self.output_dir}) already exists and is not empty. "
#                     "Use --overwrite_output_dir to overcome."
#                 )
#             elif self.last_checkpoint is not None and self.resume_from_checkpoint is None:
#                 print(
#                     f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
#                     "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
#                 )
