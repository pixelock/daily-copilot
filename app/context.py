# coding: utf-8
# @File: context.py
# @Author: pixelock
# @Time: 2023/6/11 2:34

from dataclasses import dataclass
from typing import Optional

from models.base import BaseModel


@dataclass
class Context:
    llm_model: Optional[BaseModel] = None


g = Context()
