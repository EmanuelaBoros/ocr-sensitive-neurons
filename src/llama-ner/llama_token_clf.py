# -*- coding: utf-8 -*-

import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

from modeling_llama import LlamaForTokenClassification
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
import pprint


def load_ontonotesv5():
    ret = {}
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(f"./data/NER/ontonotesv5/{split_name}.jsonl", "r") as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)


import json
from collections import defaultdict


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
    ret = {}
    label_set = set()  # Set to collect all unique labels
    for split_name in ["train", "dev", "test"]:
        data = []
        with open(
            f"../data/HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-{split_name}-fr.tsv",
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
    label2id = {label: idx for idx, label in enumerate(label_set)}
    for split_name in ["train", "dev", "test"]:
        for idx, sentence in enumerate(ret[split_name]):
            for i in range(1, 6):
                ret[split_name][idx][f"ner_tags{i}"] = [
                    label2id[label] for label in sentence[f"ner_tags{i}"]
                ]
        ret[split_name] = Dataset.from_list(ret[split_name])
    return (
        DatasetDict(ret),
        label2id,
    )  # Adjust DatasetDict according to your data handling


if len(sys.argv) != 3:
    print("usage python %.py task model_size")
    sys.exit()

task, model_size = sys.argv[1], sys.argv[2].lower()
print(f"handling task {task}")

epochs = 10
batch_size = 8
learning_rate = 1e-4
max_length = 64
if model_size == "7b":
    model_id = "meta-llama/Llama-2-7b-hf"
    model_id = "HuggingFaceH4/mistral-7b-grok"
    lora_r = 12
elif model_size == "13b":
    model_id = "NousResearch/Llama-2-13b-hf"
    lora_r = 12
else:
    raise NotImplementedError
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
seqeval = evaluate.load("seqeval")
# seqeval = evaluate.load("seqeval.py")
# import seqeval

if task == "wnut_17":
    ds = load_dataset("wnut_17")
    label2id = {
        "O": 0,
        "B-corporation": 1,
        "I-corporation": 2,
        "B-creative-work": 3,
        "I-creative-work": 4,
        "B-group": 5,
        "I-group": 6,
        "B-location": 7,
        "I-location": 8,
        "B-person": 9,
        "I-person": 10,
        "B-product": 11,
        "I-product": 12,
    }
elif task == "conll2003":
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
elif task == "hipe2020":
    ds, label2id = load_hipe()
    print(label2id)
elif task == "ontonotesv5":
    ds = load_ontonotesv5()
    label2id = {
        "O": 0,
        "B-NORP": 1,
        "B-PERSON": 2,
        "B-WORK_OF_ART": 3,
        "B-QUANTITY": 4,
        "B-EVENT": 5,
        "B-DATE": 6,
        "B-TIME": 7,
        "B-PERCENT": 8,
        "B-LANGUAGE": 9,
        "B-ORG": 10,
        "B-CARDINAL": 11,
        "B-LAW": 12,
        "B-GPE": 13,
        "B-PRODUCT": 14,
        "B-LOC": 15,
        "B-MONEY": 16,
        "B-ORDINAL": 17,
        "B-FAC": 18,
    }
else:
    raise NotImplementedError
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())  # ds["train"].features[f"ner_tags"].feature.names

model = LlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
).bfloat16()
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=lora_r,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_and_align_labels(examples):
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


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


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
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="my_awesome_ds_model",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
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

trainer.train()
