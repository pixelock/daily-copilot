# coding: utf-8
# @File: train.py
# @Author: pixelock
# @Time: 2023/6/13 21:16

from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from logger import logger
from augments.data import DataTrainingArguments
from augments.models import ModelArguments
from augments.training import FinetuningArguments
from utils.dataset import prepare_data, preprocess_data, DataCollatorForChatGLM
from utils.trainer import Seq2SeqTrainerForChatGLM


def train():
    t_parser = HfArgumentParser(ModelArguments)
    t_model_args, _ = t_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if t_model_args.model_type == 'chatglm':
        from augments.models import ChatGLMArguments
        T_ModelConfig = ChatGLMArguments
    else:
        raise ValueError(f'unknown model type: {t_model_args.model_type}')

    parser = HfArgumentParser([T_ModelConfig, DataTrainingArguments, FinetuningArguments, Seq2SeqTrainingArguments])
    model_args, data_args, finetuning_args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True,
    )

    dataset = prepare_data(data_args=data_args, model_args=model_args)

    if training_args.model_type == 'chatglm':
        from models.chatglm import ChatGLM
        model = ChatGLM.from_pretrained(model_args=model_args, finetuning_args=finetuning_args)

    dataset = preprocess_data(
        dataset=dataset,
        tokenizer=model.tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    data_collator = DataCollatorForChatGLM(
        model=model.model,
        tokenizer=model.tokenizer,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
    )

    if data_args.eval_ratio:
        dataset = dataset.train_test_split(test_size=data_args.eval_ratio)
        trainer_kwargs = {
            'train_dataset': dataset['train'],
            'eval_dataset': dataset['test'],
        }
    else:
        trainer_kwargs = {
            'train_dataset': dataset,
        }

    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model.model,
        tokenizer=model.tokenizer,
        args=training_args,
        data_collator=data_collator,
        **trainer_kwargs,
    )

    train_results = trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    train()
