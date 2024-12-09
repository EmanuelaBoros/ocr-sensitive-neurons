# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from modeling_mistral import MistralForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import pickle

import argparse
import json
from collections import defaultdict
from modeling_llama import LlamaForTokenClassification
import logging
import os, csv

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
import pprint


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


seqeval = evaluate.load("seqeval")


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


# task, model_size = "hipe2020", "7b"


def tokenize_and_align_labels(tokenizer, examples, max_length=256):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="longest",
        max_length=max_length,
        truncation=True,
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags1"]):
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
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


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
        "--layer_level",
        type=str,
        default="First_Layers",
        choices=["First_Layers", "Middle_Layers", "Last_Layers"],
        help="Layer level to dampen",
    )
    parser.add_argument(
        "--num_neurons",
        type=int,
        default=1000,
        help="Number of neurons to temper in the specified layers",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum length of the sentences",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="Layer to temper with",
    )
    parser.add_argument(
        "--decrease",
        type=float,
        default=0.1,
        help="Decrease factor for neuron tempering",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="llama or mistral",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If specified, loads the last checkpoint and evaluates on the test set",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="trained_models",
        help="The directory where the model outputs will be saved.",
    )
    args = parser.parse_args()

    # Configure the output directory based on the layer level and number of neurons
    # output_directory = f"trained_models/finetune_{args.layer_level.lower()}_{args.num_neurons}_{args.decrease}"
    output_directory = f"{args.output_directory}/test_{args.layer}_{args.num_neurons}_{args.decrease}_{args.task}"
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
    lora_r = 12

    if args.test:
        if os.path.exists(f"{output_directory}/test_metrics.log"):
            print(f"Test metrics already exist for {output_directory}. Exiting...")
            sys.exit(0)
        if args.model == "llama":
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            tokenizer.pad_token = tokenizer.eos_token

            model = LlamaForTokenClassification.from_pretrained(
                "NousResearch/Llama-2-7b-hf",
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                target_indices=[args.layer],
                num_neurons=args.num_neurons,
                decrease=args.decrease,
                device_map="auto",
            ).bfloat16()
            if "ajmc" in task:
                adapter_path = "trained_models/test_-1_-1_-1.0_ajmc/checkpoint-56/"
            elif "newseye" in task:
                adapter_path = "trained_models/test_2_-1_-1.0_newseye/checkpoint-1335/"
            else:
                adapter_path = "trained_models/test_-1_0_-1.0/checkpoint-355/"
        elif args.model == "mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
            tokenizer.pad_token = tokenizer.eos_token
            model = MistralForTokenClassification.from_pretrained(
                "mistralai/Mistral-7B-v0.3",
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                target_indices=[args.layer],
                num_neurons=args.num_neurons,
                decrease=args.decrease,
                device_map="auto",
            ).bfloat16()
            if "ajmc" in task:
                adapter_path = (
                    "trained_models_mistral/test_-1_-1_-1.0_ajmc/checkpoint-56"
                )
            elif "hipe2020" in task:
                adapter_path = (
                    "trained_models_mistral/test_-1_-1_-1.0_hipe2020/checkpoint-355"
                )
            else:
                adapter_path = "trained_models/test_-1_0_-1.0/checkpoint-355/"
        print(f"Loading the latest checkpoint: {adapter_path}")
        adapter_name = "latest_adapter"  # You can choose a name for your adapter
        model.load_adapter(adapter_path, adapter_name)
        print(model)
    else:
        if args.model == "llama":
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            tokenizer.pad_token = tokenizer.eos_token
            model = LlamaForTokenClassification.from_pretrained(
                "NousResearch/Llama-2-7b-hf",
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                target_indices=[args.layer_level],
                num_neurons=args.num_neurons,
                device_map="auto",
            ).bfloat16()
        elif args.model == "mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
            tokenizer.pad_token = tokenizer.eos_token
            model = MistralForTokenClassification.from_pretrained(
                "mistralai/Mistral-7B-v0.3",
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                target_indices=[args.layer_level],
                num_neurons=args.num_neurons,
                device_map="auto",
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

    tokenized_ds = ds.map(
        lambda examples: tokenize_and_align_labels(
            tokenizer, examples, max_length=args.max_length
        ),
        batched=True,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    setup_logging(output_directory)

    training_args = TrainingArguments(
        output_dir=output_directory,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=53,
        save_strategy="steps",
        save_steps=53,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

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

