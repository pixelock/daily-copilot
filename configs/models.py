# coding: utf-8
# @File: models.py
# @Author: pixelock
# @Time: 2023/6/13 20:32

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                    "dtype will be automatically derived from the model's weights.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    model_type: Optional[str] = None


@dataclass
class ChatGLMConfig(ModelConfig):
    temperature: float = 0.75
    top_p: float = 0.9
    max_tokens: int = 2048
    pre_seq_len: Optional[int] = None

    def __post_init__(self):
        self.model_type = 'ChatGLM'
