from utils_ner import write_predictions
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModel, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import json
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from seqeval.metrics import classification_report as seq_classification_report
from torch import nn
from tqdm import tqdm, trange
import os
import logging
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
from typing import List, Optional, Tuple, Union

from transformers.models.llama.modeling_llama import (
    SequenceClassifierOutputWithPast,
    add_start_docstrings_to_model_forward,
)

# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
# add the current directory to sys.path
sys.path.insert(0, current_directory)
from utils_ner import set_seed, SEED

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score, precision_score, recall_score
from modeling_llama import LlamaPreTrainedModel, LlamaModel, LLAMA_INPUTS_DOCSTRING


def _move_model_to_device(model, device):
    """Move a model to a device (CPU or GPU)."""
    model = model.to(device)
    # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
    # model.tie_weights()
    return model


class ModelForSequenceAndTokenClassification(PreTrainedModel):

    def __init__(self, config, num_coarse_token_labels, num_fine_token_labels):
        super().__init__(config)
        self.num_coarse_token_labels = num_coarse_token_labels
        self.num_fine_token_labels = num_fine_token_labels
        # self.num_sequence_labels = num_sequence_labels
        self.config = config

        self.bert = AutoModel.from_config(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # For token classification
        self.token_coarse_classifier = nn.Linear(
            config.hidden_size, self.num_coarse_token_labels
        )
        self.token_fine_classifier = nn.Linear(
            config.hidden_size, self.num_fine_token_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_coarse_targets: Optional[torch.Tensor] = None,
        token_fine_targets: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # For token classification
        token_output = outputs[0]

        token_output = self.dropout(token_output)

        token_coarse_token_logits = self.token_coarse_classifier(token_output)
        token_fine_token_logits = self.token_fine_classifier(token_output)

        # Computing the loss as the average of both losses
        loss = None
        loss_coarse_tokens = None
        loss_fine_tokens = None
        if token_coarse_targets is not None:
            loss_fct = CrossEntropyLoss()
            loss_coarse_tokens = loss_fct(
                token_coarse_token_logits.view(-1, self.num_coarse_token_labels),
                token_coarse_targets.view(-1),
            )

            loss_fine_tokens = loss_fct(
                token_fine_token_logits.view(-1, self.num_fine_token_labels),
                token_fine_targets.view(-1),
            )

            loss = loss_coarse_tokens + loss_fine_tokens

        if not return_dict:
            output = (token_coarse_token_logits, token_fine_token_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss_coarse_tokens,
            logits=token_coarse_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), TokenClassifierOutput(
            loss=loss_fine_tokens,
            logits=token_fine_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config, num_coarse_token_labels, num_fine_token_labels):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.dropout = nn.Dropout(0.1)

        self.num_coarse_token_labels = num_coarse_token_labels
        self.num_fine_token_labels = num_fine_token_labels

        self.token_coarse_classifier = nn.Linear(
            config.hidden_size, self.num_coarse_token_labels
        )
        self.token_fine_classifier = nn.Linear(
            config.hidden_size, self.num_fine_token_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_coarse_targets: Optional[torch.Tensor] = None,
        token_fine_targets: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # For token classification
        token_output = outputs[0]

        token_output = self.dropout(token_output)

        token_coarse_token_logits = self.token_coarse_classifier(token_output)
        token_fine_token_logits = self.token_fine_classifier(token_output)

        # Computing the loss as the average of both losses
        loss = None
        loss_coarse_tokens = None
        loss_fine_tokens = None
        if token_coarse_targets is not None:
            loss_fct = CrossEntropyLoss()
            loss_coarse_tokens = loss_fct(
                token_coarse_token_logits.view(-1, self.num_coarse_token_labels),
                token_coarse_targets.view(-1),
            )

            loss_fine_tokens = loss_fct(
                token_fine_token_logits.view(-1, self.num_fine_token_labels),
                token_fine_targets.view(-1),
            )

            loss = loss_coarse_tokens + loss_fine_tokens

        if not return_dict:
            output = (token_coarse_token_logits, token_fine_token_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss_coarse_tokens,
            logits=token_coarse_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), TokenClassifierOutput(
            loss=loss_fine_tokens,
            logits=token_fine_token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def evaluate(
    args,
    model,
    dataset,
    nerc_coarse_label_map,
    nerc_fine_label_map,
    prefix="",
    tokenizer=None,
):
    # eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(dataset)
        if args.local_rank == -1
        else DistributedSampler(dataset)
    )
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_token_fine_ids, out_token_coarse_ids = None, None
    out_token_fine_preds, out_token_coarse_preds = None, None
    sentences, text_sentences = None, None
    offset_mappings = None
    model.eval()

    # finish = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # batch = tuple(t.to(args.device) for t in batch)

        sequence = batch["sequence"]

        with torch.no_grad():
            # logger.info("Model type: %s", args.model_type.lower())
            if "llama" in args.model_type.lower():
                # inputs["token_type_ids"] = (
                #     batch[2] if args.model_type in ["bert", "xlnet"] else None
                # )  # XLM and RoBERTa don"t use segment_ids
                del batch["token_type_ids"]

                del batch["sequence"]

            model = _move_model_to_device(model, args.device)
            token_coarse_result, token_fine_result = model(**batch)

            # tokens_result = outputs[0], outputs[1]
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss_coarse = token_coarse_result.loss
            loss_fine = token_fine_result.loss

            loss = loss_coarse + loss_fine

            # tokens_result = outputs[0], outputs[1]
            token_coarse_logits = token_coarse_result.logits
            token_fine_logits = token_coarse_result.logits

            # the second return value is logits
            # sequence_logits = sequence_result.logits

            tmp_eval_loss = loss

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # import pdb;
        # pdb.set_trace()
        if out_token_coarse_preds is None:

            out_token_coarse_preds = token_coarse_logits.detach().cpu()
            out_token_coarse_preds = torch.argmax(
                out_token_coarse_preds, axis=2
            ).numpy()

            out_token_fine_preds = token_fine_logits.detach().cpu()
            out_token_fine_preds = torch.argmax(out_token_fine_preds, axis=2).numpy()

            out_token_coarse_ids = batch["token_coarse_targets"].detach().cpu()
            out_token_fine_ids = batch["token_fine_targets"].detach().cpu()

            sentences = [
                tokenizer.convert_ids_to_tokens(input_ids)
                for input_ids in batch["input_ids"].detach().cpu()
            ]
            text_sentences = [text.split(" ") for text in sequence]

        else:
            token_coarse_logits = token_coarse_logits.detach().cpu()
            token_coarse_logits = torch.argmax(token_coarse_logits, axis=2).numpy()
            out_token_coarse_preds = np.append(
                out_token_coarse_preds,
                token_coarse_logits,
                axis=0,
            )
            token_fine_logits = token_fine_logits.detach().cpu()
            token_fine_logits = torch.argmax(token_fine_logits, axis=2).numpy()
            out_token_fine_preds = np.append(
                out_token_fine_preds, token_fine_logits, axis=0
            )

            out_token_coarse_ids = torch.cat(
                (
                    out_token_coarse_ids,
                    batch["token_coarse_targets"].detach().cpu(),
                )
            )
            out_token_fine_ids = torch.cat(
                (out_token_fine_ids, batch["token_fine_targets"].detach().cpu())
            )

            sentences = np.append(
                sentences,
                [
                    tokenizer.convert_ids_to_tokens(input_ids)
                    for input_ids in batch["input_ids"].detach().cpu()
                ],
                axis=0,
            )

            # text_sentences = np.append(text_sentences, [text.split(' ') for text in batch["sequence"]], axis=0)
            text_sentences = text_sentences + [text.split(" ") for text in sequence]

    # out_token_coarse_preds = np.argmax(out_token_coarse_preds, axis=2)
    # out_token_fine_preds = np.argmax(out_token_fine_preds, axis=2)

    eval_loss = eval_loss / nb_eval_steps

    nerc_coarse_label_map = {label: i for i, label in nerc_coarse_label_map.items()}
    nerc_fine_label_map = {label: i for i, label in nerc_fine_label_map.items()}

    out_coarse_label_list = [[] for _ in range(out_token_coarse_ids.shape[0])]
    out_fine_label_list = [[] for _ in range(out_token_coarse_ids.shape[0])]
    preds_coarse_list = [[] for _ in range(out_token_coarse_ids.shape[0])]
    preds_fine_list = [[] for _ in range(out_token_coarse_ids.shape[0])]
    words_list = [[] for _ in range(out_token_coarse_ids.shape[0])]

    for idx_sentence, item in enumerate(
        zip(
            text_sentences,
            out_token_coarse_ids,
            out_token_fine_ids,
            out_token_coarse_preds,
            out_token_fine_preds,
        )
    ):
        (
            text_sentence,
            out_label_coarse_ids,
            out_label_fine_ids,
            out_label_coarse_preds,
            out_label_fine_preds,
        ) = item
        word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
        for idx, word in enumerate(text_sentence):
            beginning_index = word_ids.index(idx)
            try:
                out_coarse_label_list[idx_sentence].append(
                    nerc_coarse_label_map[out_label_coarse_ids[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                out_coarse_label_list[idx_sentence].append("O")
            try:
                preds_coarse_list[idx_sentence].append(
                    nerc_coarse_label_map[out_label_coarse_preds[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                preds_coarse_list[idx_sentence].append("O")

            # ------- fine
            try:
                out_fine_label_list[idx_sentence].append(
                    nerc_fine_label_map[out_label_fine_ids[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                out_fine_label_list[idx_sentence].append("O")
            try:
                preds_fine_list[idx_sentence].append(
                    nerc_fine_label_map[out_label_fine_preds[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                preds_fine_list[idx_sentence].append("O")

            words_list[idx_sentence].append(word)

    results = {
        "loss": eval_loss,
        "precision-coarse": precision_score(out_coarse_label_list, preds_coarse_list),
        "recall-coarse": recall_score(out_coarse_label_list, preds_coarse_list),
        "f1-coarse": f1_score(out_coarse_label_list, preds_coarse_list),
        "precision-fine": precision_score(out_fine_label_list, preds_fine_list),
        "recall-fine": recall_score(out_fine_label_list, preds_fine_list),
        "f1-fine": f1_score(out_fine_label_list, preds_fine_list),
    }

    logger.info("Evaluation for named entity recognition & classification - coarse.")
    report_class_coarse = seq_classification_report(
        out_coarse_label_list, preds_coarse_list, digits=4
    )
    logger.info("\n%s", report_class_coarse)

    logger.info("Evaluation for named entity recognition & classification - fine.")
    report_class_fine = seq_classification_report(
        out_fine_label_list, preds_fine_list, digits=4
    )
    logger.info("\n%s", report_class_fine)

    logger.info("Evaluation for named entity recognition classification.")
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return (
        results,
        words_list,
        preds_coarse_list,
        preds_fine_list,
        report_class_coarse,
        report_class_fine,
    )


def train(
    args,
    train_dataset,
    dev_dataset,
    test_dataset,
    model,
    tokenizer,
    optimizer,
    nerc_coarse_label_map,
    nerc_fine_label_map,
):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    t_total = (
        math.ceil(len(train_dataset) / args.train_batch_size) * args.epochs
    )  # assume 10 epochs
    # 10% of training steps for warmup
    num_warmup_steps = math.ceil(t_total * 0.1)

    # Prepare optimizer and schedule (linear warmup and decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_type, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_type, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_type, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_type, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[1]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_type):
        # set global_step to global_step of last saved checkpoint from model
        # path
        global_step = int(args.model_type.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(SEED)  # Added here for reproductibility

    results_devset = {
        result: [] for result in ["global", "token-level-coarse", "token-level-fine"]
    }
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # import pdb;pdb.set_trace()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # inputs = {
            #     "input_ids": batch["input_ids"].to(args.device),
            #     "attention_mask": batch["attention_mask"].to(args.device),
            #     "token_coarse_targets": batch["token_coarse_targets"].to(args.device),
            #     "token_fine_targets": batch["token_fine_targets"].to(args.device),
            #     "token_type_ids": batch["token_type_ids"].to(args.device),
            # }
            # logger.info("Model type: %s", args.model_type.lower())
            if "llama" in args.model_type.lower():
                # inputs["token_type_ids"] = (
                #     batch[2] if args.model_type in ["bert", "xlnet"] else None
                # )  # XLM and RoBERTa don"t use segment_ids
                del batch["token_type_ids"]
                del batch["sequence"]

            model = _move_model_to_device(model, args.device)
            token_coarse_result, token_fine_result = model(**batch)

            # tokens_result = outputs[0], outputs[1]
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss_coarse = token_coarse_result.loss
            loss_fine = token_fine_result.loss

            loss = loss_coarse + loss_fine

            # import pdb;pdb.set_trace()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if args.local_rank == -1 and args.evaluate_during_training:

                        (
                            results,
                            words_list,
                            preds_list_coarse,
                            preds_list_fine,
                            report_class_coarse,
                            report_class_fine,
                        ) = evaluate(
                            args,
                            model,
                            dev_dataset,
                            nerc_coarse_label_map,
                            nerc_fine_label_map,
                            tokenizer=tokenizer,
                        )

                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )

                        results_devset["global"].append(results)
                        results_devset["token-level-coarse"].append(report_class_coarse)
                        results_devset["token-level-fine"].append(report_class_fine)

                        write_predictions(
                            args.output_dir,
                            dev_dataset.get_filename(),
                            words_list,
                            preds_list_coarse,
                            preds_list_fine,
                        )

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    (
        results,
        words_list,
        preds_list_coarse,
        preds_list_fine,
        report_class_coarse,
        report_class_fine,
    ) = evaluate(
        args,
        model,
        test_dataset,
        nerc_coarse_label_map,
        nerc_fine_label_map,
        tokenizer=tokenizer,
    )

    write_predictions(
        args.output_dir,
        test_dataset.get_filename(),
        words_list,
        preds_list_coarse,
        preds_list_fine,
    )

    results_testset = {}
    results_testset["global"] = results
    results_testset["token-level-coarse"] = report_class_coarse
    results_testset["token-level-fine"] = report_class_fine

    all_results = {"dev": results_devset, "test": results_testset}
    if "-de" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
            json.dump(all_results, f)
    elif "-fr" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
            json.dump(all_results, f)
    else:
        logger.info(
            f"Was not able to deduct language from filename of testset, thus no metrics were saved."
        )

    return global_step, tr_loss / global_step
