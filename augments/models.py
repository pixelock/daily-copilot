# coding: utf-8
# @File: models.py
# @Author: pixelock
# @Time: 2023/6/13 20:32

from typing import Optional
from dataclasses import dataclass, field
import torch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization.Don't set if you want to train a model "
                    "from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # Tokenizer
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )

    # Quantization
    quant_bit: Optional[str] = field(
        default=None,
        metadata={
            'help': 'model quantization bit',
            'choices': ['int8', 'int4'],
        }
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Used in quantization configs. Do not specify this argument manually."}
    )
    double_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )
    quant_type_4bit: Optional[str] = field(
        default='nf4',
        metadata={
            "help": "Quantization data type to use in int4 training.",
            'choices': ['fp4', 'nf4']
        }
    )

    # resume adapter
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={
            'help': 'Whether to resume training from the last LoRA weights or create new weights after merging them.'}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the directory containing the model checkpoints as well as the configurations.'}
    )

    model_type: Optional[str] = None


@dataclass
class ChatGLMArguments(ModelArguments):
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 2048

    def __post_init__(self):
        self.model_type = 'chatglm'
