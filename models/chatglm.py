# coding: utf-8
# @File: chatglm.py
# @Author: pixelock
# @Time: 2023/6/10 11:01

from typing import Dict, Optional, Any, List
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from models.base import BaseModel
from configs.models import ChatGLMConfig
from configs.training import FinetuningConfig
from configs.cuda import NUM_GPU
from utils.cuda import fetch_available_gpus, check_gpu_status

CHATGLM_GPU_MEMORY_MAP = {
    'float16': 15000,
    'int8': 9000,
    'int4': 6000,
}


def load_model_on_multi_gpus(checkpoint_path: str,
                             quantization: str = 'int8',
                             device_map: Dict = None,
                             verbose: bool = True,
                             **kwargs) -> Module:
    available_gpus = fetch_available_gpus(
        threshold=CHATGLM_GPU_MEMORY_MAP.get(quantization, 0),
        devices=device_map,
        verbose=verbose
    )
    main_device = available_gpus[0]
    if verbose:
        print(f'available gpus: {available_gpus} | main device(with largest free memory): {main_device}')

    num_gpus = len(available_gpus)
    device_map = auto_configure_device_map(device_ids=available_gpus)
    if verbose:
        print(f'device map of model parameters:\n{device_map}')

    from accelerate import dispatch_model

    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()
    model = dispatch_model(model, device_map=device_map)
    if verbose:
        print('GPUs status after model been loaded:')
        check_gpu_status(verbose=verbose)
    return model


def auto_configure_device_map(num_gpus: int = None, device_ids: List[int] = None) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_ids = device_ids or list(range(num_gpus))
    target = device_ids.pop(0)

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {
        'transformer.word_embeddings': target,
        'transformer.final_layernorm': target,
        'lm_head': target,
    }

    used = 2
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            target += 1
            used = 0
        device_map[f'transformer.layers.{i}'] = target
        used += 1

    return device_map


class ChatGLM(BaseModel):
    def prepare_env(self):
        model_kwargs = {
            'trust_remote_code': True,
        }

        if self.model_args.quant_bit is not None:
            model_kwargs['device_map'] = 'auto'

            if self.model_args.quant_bit == 'int8':
                model_kwargs['load_in_8bit'] = True
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif self.model_args.quant_bit == 'int4':
                model_kwargs['load_in_4bit'] = True
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.model_args.compute_dtype,
                    bnb_4bit_use_double_quant=self.model_args.double_quant,
                    bnb_4bit_quant_type=self.model_args.quant_type_4bit,
                )

        self.model_kwargs = model_kwargs

    def prepare_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            padding_side='left',
        )
        return tokenizer

    def prepare_model_for_training(self):
        model = AutoModel(
            self.model_args.model_name_or_path,
            **self.model_kwargs,
        )


    # def init_model(self, **kwargs):
    #     if self.do_train:
    #         if self.training_config.use_quant:
    #             if self.training_config.quant_bit == 'int8':
    #                 model = AutoModel.from_pretrained(
    #                     self.model_name_or_path,
    #                     load_in_8bit=True,
    #                     device_map="auto",
    #                     trust_remote_code=True,
    #                 )
    #                 model.use_cache = False
    #                 model.gradient_checkpointing_enable()
    #             else:
    #                 raise ValueError(f'unsupported quant bit: {self.training_config.quant_bit}')
    #         else:
    #             model = AutoModel.from_pretrained(
    #                 self.model_name_or_path,
    #                 trust_remote_code=True,
    #             )
    #
    #         if self.training_config.use_lora:
    #             model = self.get_lora_model(model)
    #     else:
    #         if self.config.device == 'cpu':
    #             model = AutoModel.from_pretrained(
    #                 self.model_name_or_path,
    #                 trust_remote_code=True,
    #                 **kwargs,
    #             ).float()
    #         elif self.config.multi_gpu and NUM_GPU > 1:
    #             model = load_model_on_multi_gpus(
    #                 checkpoint_path=self.model_name_or_path,
    #                 quantization=self.config.quant_bit if self.config.use_quant else 'float16',
    #                 verbose=True,
    #                 **kwargs,
    #             )
    #         else:
    #             model = AutoModel.from_pretrained(
    #                 self.model_name_or_path,
    #                 trust_remote_code=True,
    #                 **kwargs,
    #             ).half().to(self.config.device)
    #
    #         model.eval()
    #
    #     return model
