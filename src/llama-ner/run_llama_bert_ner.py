# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import pickle
import torch


# Configure logging
def setup_logging(output_dir):
    """Sets up logging to store logs in the specified output directory."""
    print(f"Logging to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it does not exist

    log_file = f"{output_dir}/training_metrics.log"
    print(f"Logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",  # Ensure the logs are appended to the file
    )

    # Optionally, return the logger
    logger = logging.getLogger()
    return logger


import argparse
import json
from collections import defaultdict
from modeling_llama import LlamaForTokenClassification
from modelling_bidirectional_llama import LlamaBertForTokenClassification
import logging
import os, csv

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
import pprint

from transformers import Trainer
import torch


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss using the dual tokenized inputs (BERT and LLaMA).
        """
        # Separate inputs for BERT and LLaMA
        input_ids_bert = inputs.pop("input_ids_bert")
        attention_mask_bert = inputs.pop("attention_mask_bert")
        input_ids_llama = inputs.pop("input_ids_llama")
        attention_mask_llama = inputs.pop("attention_mask_llama")
        labels = inputs.pop("labels")

        # Forward pass through the model
        outputs = model(
            input_ids_bert=input_ids_bert,
            attention_mask_bert=attention_mask_bert,
            input_ids_llama=input_ids_llama,
            attention_mask_llama=attention_mask_llama,
            labels=labels,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs, return_outputs=False)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step on a batch of inputs.
        """
        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs
        labels = inputs.get("labels") if has_labels else None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs.logits
            else:
                loss = None
                logits = model(**inputs).logits

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and return metrics.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=self.args.prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(output.metrics)
        return output.metrics

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Run predictions and return predictions and metrics.
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self.prediction_loop(
            test_dataloader,
            description="Prediction",
            prediction_loss_only=self.args.prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


def load_ontonotesv5():
    ret = {}
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(f"./data/NER/ontonotesv5/{split_name}.jsonl", "r") as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)


def log_metrics_to_csv(metrics, file_path):
    """Log metrics to a CSV file. Append if file exists, create otherwise."""
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as csvfile:
        fieldnames = ["precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the metric values
        writer.writerow(metrics)


def load_newseye():
    """
    Load HIPE 2020 dataset

    {'annotations': ['B-pers', 'O', 'B-pers.ind', 'O', 'O'], 'token': 'Etienne'}

    CoNLL:
    {'id': '13408',
    'tokens': ['Egypt', 'police', 'catch', 'ancient', 'manuscript', 'thieves', '.'],
    'pos_tags': [22, 21, 21, 16, 21, 24, 7], 'chunk_tags': [11, 12, 12, 12, 12, 12, 0],
    'ner_tags': [5, 0, 0, 0, 0, 0, 0]}

    :return:
    """
    ret_file = os.path.join("trained_models", "newseye_ret.pkl")
    label2id_file = os.path.join("trained_models", "newseye_label2id.pkl")

    # Check if the files already exist
    if os.path.exists(ret_file) and os.path.exists(label2id_file):
        with open(ret_file, "rb") as f:
            ret = pickle.load(f)
        with open(label2id_file, "rb") as f:
            label2id = pickle.load(f)
        return DatasetDict(ret), label2id

    ret = {}
    label_set = set()  # Set to collect all unique labels
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(
            f"data/newseye/fr/HIPE-2022-v2.1-newseye-{split_name}-fr.tsv",
            "r",
            encoding="utf-8",
        ) as reader:
            document = {}
            tokens = []
            annotations = {}
            for i in range(1, 6):
                annotations[f"ner_tags{i}"] = []
            for line in reader:
                if "NE-COARSE-LIT" in line:
                    columns = line.strip().split("\t")
                    continue
                if line.startswith("# "):  # Metadata or comments
                    if line.startswith("# hipe2022:document_id"):
                        if "doc_id" not in document:
                            document = {
                                "doc_id": line.strip().split("=")[1].strip(),
                            }

                    elif line.strip().startswith("# "):
                        key, value = line.strip().split("=", 1)
                        if "metadata" not in document:
                            document["metadata"] = {}
                        document["metadata"][key.strip()] = value.strip()
                else:
                    parts = line.strip().split("\t")
                    if (line.strip() == "") or ("EndOfSentence" in line):
                        if tokens:
                            sentence = {"tokens": tokens}
                            for i in range(1, 6):
                                sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                            data.append(sentence)
                            tokens = []
                            annotations = {}
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"] = []
                    else:
                        tokens.append(parts[0])
                        if len(parts) > 6:
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"].append(parts[i])
                                if parts[i] != "_":  # Assuming "_" means no label
                                    label_set.add(parts[i])

            if tokens:  # Catch last sentence
                sentence = {"tokens": tokens}
                for i in range(1, 6):
                    sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                data.append(sentence)
                annotations = {}
                for i in range(1, 6):
                    annotations[f"ner_tags{i}"] = []
        ret[split_name] = data

    # Generate label2id dictionary
    print(label_set)
    label2id = {label: idx for idx, label in enumerate(label_set)}
    for split_name in ["train", "dev", "test"]:
        for idx, sentence in enumerate(ret[split_name]):
            for i in range(1, 6):
                ret[split_name][idx][f"ner_tags{i}"] = [
                    label2id[label]
                    for label in sentence[f"ner_tags{i}"]
                    if label != "_"
                ]
        ret[split_name] = Dataset.from_list(ret[split_name])

    # Save the processed data and label2id dictionary
    with open(ret_file, "wb") as f:
        pickle.dump(ret, f)
    with open(label2id_file, "wb") as f:
        pickle.dump(label2id, f)

    return DatasetDict(ret), label2id


def load_ajmc():
    """
    Load HIPE 2020 dataset

    {'annotations': ['B-pers', 'O', 'B-pers.ind', 'O', 'O'], 'token': 'Etienne'}

    CoNLL:
    {'id': '13408',
    'tokens': ['Egypt', 'police', 'catch', 'ancient', 'manuscript', 'thieves', '.'],
    'pos_tags': [22, 21, 21, 16, 21, 24, 7], 'chunk_tags': [11, 12, 12, 12, 12, 12, 0],
    'ner_tags': [5, 0, 0, 0, 0, 0, 0]}

    :return:
    """
    ret_file = os.path.join("trained_models", "ajmc_ret.pkl")
    label2id_file = os.path.join("trained_models", "ajmc_label2id.pkl")

    # Check if the files already exist
    if os.path.exists(ret_file) and os.path.exists(label2id_file):
        with open(ret_file, "rb") as f:
            ret = pickle.load(f)
        with open(label2id_file, "rb") as f:
            label2id = pickle.load(f)
        return DatasetDict(ret), label2id

    ret = {}
    label_set = set()  # Set to collect all unique labels
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(
            f"data/ajmc/fr/HIPE-2022-v2.1-ajmc-{split_name}-fr.tsv",
            "r",
            encoding="utf-8",
        ) as reader:
            document = {}
            tokens = []
            annotations = {}
            for i in range(1, 6):
                annotations[f"ner_tags{i}"] = []
            for line in reader:
                if "NE-COARSE-LIT" in line:
                    columns = line.strip().split("\t")
                    continue
                if line.startswith("# "):  # Metadata or comments
                    if line.startswith("# hipe2022:document_id"):
                        if "doc_id" not in document:
                            document = {
                                "doc_id": line.strip().split("=")[1].strip(),
                            }

                    elif line.strip().startswith("# "):
                        key, value = line.strip().split("=", 1)
                        if "metadata" not in document:
                            document["metadata"] = {}
                        document["metadata"][key.strip()] = value.strip()
                else:
                    parts = line.strip().split("\t")
                    if (line.strip() == "") or ("EndOfSentence" in line):
                        if tokens:
                            sentence = {"tokens": tokens}
                            for i in range(1, 6):
                                sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                            data.append(sentence)
                            tokens = []
                            annotations = {}
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"] = []
                    else:
                        tokens.append(parts[0])
                        if len(parts) > 6:
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"].append(parts[i])
                                if parts[i] != "_":  # Assuming "_" means no label
                                    label_set.add(parts[i])

            if tokens:  # Catch last sentence
                sentence = {"tokens": tokens}
                for i in range(1, 6):
                    sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                data.append(sentence)
                annotations = {}
                for i in range(1, 6):
                    annotations[f"ner_tags{i}"] = []
        ret[split_name] = data

    # Generate label2id dictionary
    print(label_set)
    label2id = {label: idx for idx, label in enumerate(label_set)}
    for split_name in ["train", "dev", "test"]:
        for idx, sentence in enumerate(ret[split_name]):
            for i in range(1, 6):
                ret[split_name][idx][f"ner_tags{i}"] = [
                    label2id[label]
                    for label in sentence[f"ner_tags{i}"]
                    if label != "_"
                ]
        ret[split_name] = Dataset.from_list(ret[split_name])

    # Save the processed data and label2id dictionary
    with open(ret_file, "wb") as f:
        pickle.dump(ret, f)
    with open(label2id_file, "wb") as f:
        pickle.dump(label2id, f)

    return DatasetDict(ret), label2id


def load_hipe():
    """
    Load HIPE 2020 dataset

    {'annotations': ['B-pers', 'O', 'B-pers.ind', 'O', 'O'], 'token': 'Etienne'}

    CoNLL:
    {'id': '13408',
    'tokens': ['Egypt', 'police', 'catch', 'ancient', 'manuscript', 'thieves', '.'],
    'pos_tags': [22, 21, 21, 16, 21, 24, 7], 'chunk_tags': [11, 12, 12, 12, 12, 12, 0],
    'ner_tags': [5, 0, 0, 0, 0, 0, 0]}

    :return:
    """
    ret_file = os.path.join("trained_models", "hipe_ret.pkl")
    label2id_file = os.path.join("trained_models", "label2id.pkl")

    # Check if the files already exist
    if os.path.exists(ret_file) and os.path.exists(label2id_file):
        with open(ret_file, "rb") as f:
            ret = pickle.load(f)
        with open(label2id_file, "rb") as f:
            label2id = pickle.load(f)
        return DatasetDict(ret), label2id

    ret = {}
    label_set = set()  # Set to collect all unique labels
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(
            f"data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-{split_name}-fr.tsv",
            "r",
            encoding="utf-8",
        ) as reader:
            document = {}
            tokens = []
            annotations = {}
            for i in range(1, 6):
                annotations[f"ner_tags{i}"] = []
            for line in reader:
                if "NE-COARSE-LIT" in line:
                    columns = line.strip().split("\t")
                    continue
                if line.startswith("# "):  # Metadata or comments
                    if line.startswith("# hipe2022:document_id"):
                        document = {
                            "doc_id": line.strip().split("=")[1].strip(),
                            "metadata": {},
                        }
                    elif line.strip().startswith("# "):
                        key, value = line.strip().split("=", 1)
                        document["metadata"][key.strip()] = value.strip()
                else:
                    parts = line.strip().split("\t")
                    if (line.strip() == "") or ("EndOfSentence" in line):
                        if tokens:
                            sentence = {"tokens": tokens}
                            for i in range(1, 6):
                                sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                            data.append(sentence)
                            tokens = []
                            annotations = {}
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"] = []
                    else:
                        tokens.append(parts[0])
                        if len(parts) > 6:
                            for i in range(1, 6):
                                annotations[f"ner_tags{i}"].append(parts[i])
                                if parts[i] != "_":  # Assuming "_" means no label
                                    label_set.add(parts[i])

            if tokens:  # Catch last sentence
                sentence = {"tokens": tokens}
                for i in range(1, 6):
                    sentence[f"ner_tags{i}"] = annotations[f"ner_tags{i}"]
                data.append(sentence)
                annotations = {}
                for i in range(1, 6):
                    annotations[f"ner_tags{i}"] = []
        ret[split_name] = data

    # Generate label2id dictionary
    print(label_set)
    label2id = {label: idx for idx, label in enumerate(label_set)}
    for split_name in ["train", "dev", "test"]:
        for idx, sentence in enumerate(ret[split_name]):
            for i in range(1, 6):
                ret[split_name][idx][f"ner_tags{i}"] = [
                    label2id[label] for label in sentence[f"ner_tags{i}"]
                ]
        ret[split_name] = Dataset.from_list(ret[split_name])

    # Save the processed data and label2id dictionary
    with open(ret_file, "wb") as f:
        pickle.dump(ret, f)
    with open(label2id_file, "wb") as f:
        pickle.dump(label2id, f)

    return DatasetDict(ret), label2id


task, model_size = "hipe2020", "7b"


# learning_rate = 1e-4
# max_length = 64
if model_size == "7b":
    model_id = "meta-llama/Llama-2-7b-hf"
    lora_r = 12
elif model_size == "13b":
    model_id = "NousResearch/Llama-2-13b-hf"
    lora_r = 12
else:
    raise NotImplementedError
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
seqeval = evaluate.load("seqeval")


def tokenize_and_align_labels(
    bert_tokenizer, llama_tokenizer, examples, max_length=256
):
    # Tokenize with BERT tokenizer
    tokenized_inputs_bert = bert_tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    # Tokenize with LLaMA tokenizer
    tokenized_inputs_llama = llama_tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    def align_labels(tokenized_inputs, labels, tokenizer):
        aligned_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    try:
                        label_ids.append(label[word_idx])
                    except:
                        import pdb

                        pdb.set_trace()
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        return aligned_labels

    labels = examples["ner_tags1"]

    # Align labels for BERT tokenizer
    tokenized_inputs_bert["labels"] = align_labels(
        tokenized_inputs_bert, labels, bert_tokenizer
    )

    # Align labels for LLaMA tokenizer
    tokenized_inputs_llama["labels"] = align_labels(
        tokenized_inputs_llama, labels, llama_tokenizer
    )

    # Combine both tokenized inputs
    combined_inputs = {
        "input_ids_bert": tokenized_inputs_bert["input_ids"],
        "attention_mask_bert": tokenized_inputs_bert["attention_mask"],
        "input_ids_llama": tokenized_inputs_llama["input_ids"],
        "attention_mask_llama": tokenized_inputs_llama["attention_mask"],
        "labels": tokenized_inputs_bert[
            "labels"
        ],  # Assuming labels are the same for both tokenizers
    }

    return combined_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # logger.info(f"Results: {results}")
    # Create a PrettyPrinter instance
    pp = pprint.PrettyPrinter(indent=4)

    # Use the PrettyPrinter to display the results
    logger.info("Results:")
    pp.pprint(results)
    metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    logging.info("Metrics: %s", metrics)

    log_file = f"{output_directory}/training_metrics.log"

    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="") as csvfile:
        fieldnames = ["precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()

        # Write the metric values
        writer.writerow(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on NER with specific layer and neuron settings"
    )

    # Adding arguments
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for both training and evaluation",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--task", type=str, default="hipe2020", help="Task to train on")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum length of the sentences",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If specified, loads the last checkpoint and evaluates on the test set",
    )
    args = parser.parse_args()

    # Configure the output directory based on the layer level and number of neurons
    # output_directory = f"trained_models/finetune_{args.layer_level.lower()}_{args.num_neurons}_{args.decrease}"
    output_directory = f"trained_models_june_2024/train_{args.epochs}_{args.task}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    task = args.task
    print(f"Training on {task}")
    if task == "conll2003":
        ds = load_dataset("conll2003")
        label2id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-MISC": 7,
            "I-MISC": 8,
        }
    elif task == "newseye":
        ds, label2id = load_newseye()
        # print(label2id)
    elif task == "ajmc":
        ds, label2id = load_ajmc()
        # print(label2id)
    elif task == "hipe2020":
        ds, label2id = load_hipe()
        # print(label2id)
    else:
        raise NotImplementedError
    id2label = {v: k for k, v in label2id.items()}
    label_list = list(
        label2id.keys()
    )  # ds["train"].features[f"ner_tags"].feature.names

    if args.test:
        print("for test")
        # if os.path.exists(f"{output_directory}/test_metrics.log"):
        #     print(f"Test metrics already exist for {output_directory}. Exiting...")
        #     sys.exit(0)
        # model = LlamaForTokenClassification.from_pretrained(
        #     model_id,
        #     num_labels=len(label2id),
        #     id2label=id2label,
        #     label2id=label2id,
        #     target_indices=[args.layer],
        #     num_neurons=args.num_neurons,
        #     decrease=args.decrease,  # layer_level="First_Layers", num_neurons=1000, decrease=0.1
        # ).bfloat16()
        # if "ajmc" in task:
        #     adapter_path = "trained_models/test_-1_-1_-1.0_ajmc/checkpoint-56/"
        # elif "newseye" in task:
        #     adapter_path = "trained_models/test_2_-1_-1.0_newseye/checkpoint-1335/"
        # else:
        #     adapter_path = "trained_models/test_-1_0_-1.0/checkpoint-355/"
        # print(f"Loading the latest checkpoint: {adapter_path}")
        # adapter_name = "latest_adapter"  # You can choose a name for your adapter
        # model.load_adapter(adapter_path, adapter_name)
        # print(model)
    else:
        model = LlamaBertForTokenClassification.from_pretrained(
            model_id,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            use_bert=True,
        ).bfloat16()

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            target_modules=["up_proj"],
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(model)

    from transformers import AutoTokenizer

    # Initialize the tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Set padding tokens
    # bert_tokenizer.pad_token = bert_tokenizer.eos_token
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # Verify the padding tokens
    assert (
        llama_tokenizer.pad_token is not None
    ), "LLaMA tokenizer padding token is not set."
    # assert (
    #     bert_tokenizer.pad_token is not None
    # ), "BERT tokenizer padding token is not set."

    # Tokenize the datasets
    tokenized_ds = ds.map(
        lambda x: tokenize_and_align_labels(bert_tokenizer, llama_tokenizer, x),
        batched=True,
    )

    def combined_data_collator(features):
        batch_bert = {
            k: [dic[k] for dic in features]
            for k in features[0]
            if k.startswith("input_ids_bert") or k.startswith("attention_mask_bert")
        }
        batch_llama = {
            k: [dic[k] for dic in features]
            for k in features[0]
            if k.startswith("input_ids_llama") or k.startswith("attention_mask_llama")
        }
        labels = [dic["labels"] for dic in features]

        batch = {
            "input_ids_bert": torch.tensor(
                batch_bert["input_ids_bert"], dtype=torch.long
            ),
            "attention_mask_bert": torch.tensor(
                batch_bert["attention_mask_bert"], dtype=torch.long
            ),
            "input_ids_llama": torch.tensor(
                batch_llama["input_ids_llama"], dtype=torch.long
            ),
            "attention_mask_llama": torch.tensor(
                batch_llama["attention_mask_llama"], dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch

    # Initialize model with BERT encoder enabled
    model = LlamaBertForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        use_bert=True,
    ).bfloat16()

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        target_modules=["up_proj"],
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)

    training_args = TrainingArguments(
        output_dir=output_directory,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    # Use the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=llama_tokenizer,  # Primary tokenizer for the Trainer
        data_collator=combined_data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    if args.test:
        logger.info("Loading the best model and evaluating on the test set.")
        test_results = trainer.predict(tokenized_ds["test"])
        predictions = test_results.predictions
        label_ids = test_results.label_ids

        metrics = compute_metrics((predictions, label_ids))
        logger.info(f"Test set metrics: {metrics}")
        log_metrics_to_csv(metrics, f"{output_directory}/test_metrics.log")
    else:
        trainer.train()

    setup_logging(output_directory)

    if args.test:
        logger.info("Loading the best model and evaluating on the test set.")
        test_results = trainer.predict(tokenized_ds["test"])
        predictions = test_results.predictions
        label_ids = test_results.label_ids

        metrics = compute_metrics((predictions, label_ids))
        logger.info(f"Test set metrics: {metrics}")
        log_metrics_to_csv(metrics, f"{output_directory}/test_metrics.log")
    else:
        trainer.train()
