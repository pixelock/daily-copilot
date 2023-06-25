# coding: utf-8
# @File: trainer.py
# @Author: pixelock
# @Time: 2023/6/25 23:42

import os
import sys
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Optional
from transformers import Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME, PredictionOutput
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils import PreTrainedTokenizer
from peft.utils import WEIGHTS_NAME

from logger import logger
from augments.training import FinetuningArguments
from utils.models import get_state_dict
from utils.dataset import IGNORE_INDEX

VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(weights_file):
        logger.warning("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    model_state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(model_state_dict, strict=False)  # skip missing keys
    return True


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True


class PeftTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"):  # for models with valuehead
            backbone_model = getattr(model, "pretrained_model")
        else:
            backbone_model = model

        if hasattr(backbone_model, "peft_config"):  # peft methods
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))  # save lora weights
        else:
            torch.save(get_state_dict(backbone_model), os.path.join(output_dir, WEIGHTS_NAME))  # save trainable weights

        if hasattr(model, "v_head"):  # save valuehead weights
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        model = unwrap_model(self.model)
        if hasattr(model, "peft_config"):  # peft methods
            model.load_adapter(self.state.best_model_checkpoint, getattr(model, "active_adapter"))
        else:
            load_trainable_params(model, self.state.best_model_checkpoint)

        if hasattr(model, "v_head"):
            load_valuehead_params(model, self.state.best_model_checkpoint)


class Seq2SeqTrainerForChatGLM(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def save_predictions(
            self,
            predict_results: PredictionOutput,
            tokenizer: PreTrainedTokenizer
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions,
                         self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids,
                          self.tokenizer.pad_token_id)

        preds = [pred[(pred == self.tokenizer.bos_token_id).nonzero()[0][0]:] for pred in preds]  # remove the queries
        preds = [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in preds]
        labels = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(preds, labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
