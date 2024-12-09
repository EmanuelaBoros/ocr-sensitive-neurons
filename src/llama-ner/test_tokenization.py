from utils_ner import set_seed, SEED
from model import train, evaluate
import argparse
import torch
from model import ModelForSequenceAndTokenClassification
import os
from dataset import NewsDataset
import logging
from transformers import AutoTokenizer, AutoConfig, AdamW
import json
from utils_ner import write_predictions
import signal
import sys
import subprocess
import csv
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='bert-base-uncased',
        help="The model to be loaded. It can be a pre-trained model "
        "or a fine-tuned model (folder on the disk).")
    parser.add_argument('--train_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv train file.")
    parser.add_argument('--dev_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv dev file.")
    parser.add_argument('--test_dataset',
                        type=str,
                        default='',
                        help="Path to the *csv or *tsv test file.")
    parser.add_argument('--nerc_coarse_label_map',
                        type=str,
                        default='data/nerc_fine_label_map.json',
                        help="Path to the *json file for the label mapping.")
    parser.add_argument('--nerc_fine_label_map',
                        type=str,
                        default='data/nerc_coarse_label_map.json',
                        help="Path to the *json file for the label mapping.")
    parser.add_argument('--max_sequence_len',
                        type=int, default=64,
                        help="Maximum text length.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help="Number of epochs. Default to 3 (can be 5 - max 10)")
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.")
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help="The folder where the experiment details and the predictions should be saved.")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help="The folder with a checkpoint model to be loaded and continue training or evaluate.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates steps.")
    parser.add_argument(
        "--logging_suffix",
        type=str,
        default="",
        help="Suffix to further specify name of the folder where the logging is stored."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--max_steps",
        default=-
        1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        '--n_warmup_steps',
        type=int,
        default=0,
        help="The warmup steps - the number of steps in on epoch or 0.")
    parser.add_argument("--local_rank", type=int, default=-
                        1, help="For distributed training: local_rank")

    parser.add_argument(
        '--device',
        default='cuda',
        help="The device on which should the model run - cpu or cuda.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--continue_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to make experiment reproducible.")

    args = parser.parse_args()

    import pdb;pdb.set_trace()
    tokenizer1 = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    tokenizer2 = AutoTokenizer.from_pretrained('dbmdz/bert-base-french-europeana-cased')
    tokenizer3 = AutoTokenizer.from_pretrained('camembert-base')
    tokenizer4 = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-europeana-cased')
    tokenizer5 = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer1.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
    train_dataset = NewsDataset(
                tsv_dataset=args.train_dataset,
                tokenizer=tokenizer1,
                max_len=args.max_sequence_len,
                tokenizers=[tokenizer1, tokenizer2, tokenizer3, tokenizer4, tokenizer5])
    train_sampler = RandomSampler(
            train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    global_step = 1
    epochs_trained = global_step // (len(train_dataloader) //
                                     args.gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

    train_iterator = trange(epochs_trained, int(
        args.epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
                print(batch)