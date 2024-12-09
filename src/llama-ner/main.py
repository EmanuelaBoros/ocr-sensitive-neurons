from utils_ner import set_seed, SEED
from model import train, evaluate
import argparse
import torch
from model import ModelForSequenceAndTokenClassification, LlamaForTokenClassification
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


def signal_handler(sig, frame):
    logger.info("You pressed Ctrl+C! Stopping training and running evaluation.")
    # # Place your evaluation logic here
    # results, words_list, preds_list, report_bin, report_class = evaluate(
    #     args, model, dev_dataset, nerc_coarse_label_map, tokenizer=tokenizer)
    # write_predictions(
    #     args.output_dir,
    #     dev_dataset.get_filename(),
    #     words_list,
    #     preds_list)
    # # Add any additional evaluation steps you need
    sys.exit(0)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-uncased",
        help="The model to be loaded. It can be a pre-trained model "
        "or a fine-tuned model (folder on the disk).",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="",
        help="Path to the *csv or *tsv train file.",
    )
    parser.add_argument(
        "--dev_dataset", type=str, default="", help="Path to the *csv or *tsv dev file."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="",
        help="Path to the *csv or *tsv test file.",
    )
    parser.add_argument(
        "--nerc_coarse_label_map",
        type=str,
        default="data/nerc_fine_label_map.json",
        help="Path to the *json file for the label mapping.",
    )
    parser.add_argument(
        "--nerc_fine_label_map",
        type=str,
        default="data/nerc_coarse_label_map.json",
        help="Path to the *json file for the label mapping.",
    )
    parser.add_argument(
        "--max_sequence_len", type=int, default=64, help="Maximum text length."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs. Default to 3 (can be 5 - max 10)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The training batch size - can be changed depending on the GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The folder where the experiment details and the predictions should be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="The folder with a checkpoint model to be loaded and continue training or evaluate.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--logging_suffix",
        type=str,
        default="",
        help="Suffix to further specify name of the folder where the logging is stored.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=0,
        help="The warmup steps - the number of steps in on epoch or 0.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="The device on which should the model run - cpu or cuda.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--continue_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval or not."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to make experiment reproducible."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    args.model_type = args.model_type.lower()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(
        args.output_dir,
        "model_{}_max_sequence_length_{}_epochs_{}_run{}".format(
            args.model_type.replace("/", "_").replace("-", "_"),
            args.max_sequence_len,
            args.epochs,
            args.logging_suffix,
        ),
    )
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # logging.basicConfig(level=logging.INFO)
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "logging.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logging.info(
        "Trained models and results will be saved in {}.".format(args.output_dir)
    )

    # Setup CUDA, GPU & distributed training
    if args.device == "cpu":
        device = torch.device("cpu")
        args.n_gpu = 1
    elif args.local_rank == -1 or args.device == "cuda":
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    # if label map was not specified, generate it from data and save it in
    # output folder
    if not os.path.exists(args.nerc_coarse_label_map):
        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
        )

        nerc_coarse_label_map = train_dataset.get_nerc_coarse_label_map()
        nerc_fine_label_map = train_dataset.get_nerc_fine_label_map()

        # dataset, tokenizer, max_len, test = False, label_map = None
        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

        nerc_coarse_label_map = dev_dataset.get_nerc_coarse_label_map()
        nerc_fine_label_map = dev_dataset.get_nerc_fine_label_map()

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

        nerc_coarse_label_map = test_dataset.get_nerc_coarse_label_map()
        nerc_fine_label_map = test_dataset.get_nerc_fine_label_map()
    # if specified, load the label map and use it for the data
    else:
        nerc_coarse_label_map = json.load(open(args.nerc_coarse_label_map, "r"))
        nerc_fine_label_map = json.load(open(args.nerc_fine_label_map, "r"))

        train_dataset = NewsDataset(
            tsv_dataset=args.train_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

        dev_dataset = NewsDataset(
            tsv_dataset=args.dev_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

        test_dataset = NewsDataset(
            tsv_dataset=args.test_dataset,
            tokenizer=tokenizer,
            max_len=args.max_sequence_len,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

    if not os.path.exists(os.path.join(args.output_dir, "data")):
        os.mkdir(os.path.join(args.output_dir, "data"))

    json.dump(
        nerc_coarse_label_map,
        open(os.path.join(args.output_dir, "data/nerc_coarse_label_map.json"), "w"),
    )
    json.dump(
        nerc_fine_label_map,
        open(os.path.join(args.output_dir, "data/nerc_fine_label_map.json"), "w"),
    )

    num_coarse_token_labels, num_fine_token_labels = (
        test_dataset.get_num_coarse_token_labels(),
        test_dataset.get_num_fine_token_labels(),
    )

    logging.info(
        "Number of coarse unique token labels {}.".format(num_coarse_token_labels)
    )
    logging.info("Number of fine unique token labels {}.".format(num_fine_token_labels))
    epochs = 10
    batch_size = 8
    learning_rate = 1e-4
    max_length = 64
    model_id = "NousResearch/Llama-2-7b-hf"
    lora_r = 12
    config = AutoConfig.from_pretrained(
        model_id, problem_type="single_label_classification"
    )

    from peft import get_peft_model, LoraConfig, TaskType

    model = LlamaForTokenClassification.from_pretrained(
        model_id,
        config=config,
        num_coarse_token_labels=num_coarse_token_labels,
        num_fine_token_labels=num_fine_token_labels,
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

    # model = ModelForSequenceAndTokenClassification.from_pretrained(
    #     args.model_type,
    #     config=config,
    #     num_coarse_token_labels=num_coarse_token_labels,
    #     num_fine_token_labels=num_fine_token_labels,
    # )

    model = model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    if args.do_train:
        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )

    elif args.continue_train:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        train(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            nerc_coarse_label_map=nerc_coarse_label_map,
            nerc_fine_label_map=nerc_fine_label_map,
        )
    else:
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        # checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        config = AutoConfig.from_pretrained(
            args.checkpoint,
            problem_type="single_label_classification",
            local_files_only=True,
        )
        # LlamaForTokenClassification
        model = ModelForSequenceAndTokenClassification.from_pretrained(
            args.checkpoint,
            config=config,
            num_coarse_token_labels=num_coarse_token_labels,
            num_fine_token_labels=num_fine_token_labels,
            local_files_only=True,
        )

        model = model.to(args.device)

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, local_files_only=True
        )

    line = {
        "dataset": args.train_dataset.split("/")[-1],
        "model": "bert",
        "pre_trained_model": args.model_type,
        "max_sequence_len": args.max_sequence_len,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
    }
    # dev data
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

    write_predictions(
        args.output_dir,
        dev_dataset.get_filename(),
        words_list,
        preds_list_coarse,
        preds_list_fine,
    )
    # test data
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

    eval_script_path = "../HIPE-scorer/clef_evaluation.py"

    with open(os.path.join("experiments", "results.csv"), "a") as f_out:

        for task in ["nerc_coarse", "nerc_fine"]:

            output_file = os.path.join(
                args.output_dir,
                dev_dataset.get_filename().split("/")[-1].replace(".tsv", "_pred.tsv"),
            )

            eval_cmd = f"python {eval_script_path} --ref {dev_dataset.get_filename()} --pred {output_file} --skip-check --outdir {args.output_dir} --hipe_edition hipe-2022 --log logs --task {task}"
            subprocess.run(eval_cmd, shell=True, check=True)

            # results_path = os.path.join(args.output_dir, f"predictions_dev_{task}.tsv")
            df = pd.read_csv(output_file.replace(".tsv", f"_{task}.tsv"), sep="\t")
            print(df)

            output_file = os.path.join(
                args.output_dir,
                test_dataset.get_filename().split("/")[-1].replace(".tsv", "_pred.tsv"),
            )

            eval_script_path = "../HIPE-scorer/clef_evaluation.py"

            for task in ["nerc_coarse", "nerc_fine"]:
                eval_cmd = f"python {eval_script_path} --ref {test_dataset.get_filename()} --pred {output_file} --skip-check --outdir {args.output_dir} --hipe_edition hipe-2022 --log logs --task {task}"
                subprocess.run(eval_cmd, shell=True, check=True)

            df = pd.read_csv(output_file.replace(".tsv", f"_{task}.tsv"), sep="\t")

            # Filter the desired results
            if "coarse" in task:
                desired_filters = [
                    ("NE-COARSE-LIT-micro-fuzzy-TIME-ALL-LED-ALL", "ALL"),
                    ("NE-COARSE-LIT-micro-strict-TIME-ALL-LED-ALL", "ALL"),
                ]
            else:
                desired_filters = [
                    ("NE-FINE-LIT-micro-fuzzy-TIME-ALL-LED-ALL", "ALL"),
                    ("NE-FINE-LIT-micro-strict-TIME-ALL-LED-ALL", "ALL"),
                ]

            for system, label in desired_filters:
                filtered_row = df[
                    (df["Evaluation"] == system) & (df["Label"] == label)
                ][["Evaluation", "Label", "P", "R", "F1"]]
                print(filtered_row.to_string(index=False))
                line[
                    task + "-" + filtered_row["Evaluation"].iloc[0].split("-")[4] + "-P"
                ] = filtered_row["P"].iloc[0]
                line[
                    task + "-" + filtered_row["Evaluation"].iloc[0].split("-")[4] + "-R"
                ] = filtered_row["P"].iloc[0]
                line[
                    task
                    + "-"
                    + filtered_row["Evaluation"].iloc[0].split("-")[4]
                    + "-F1"
                ] = filtered_row["P"].iloc[0]

            writer = csv.DictWriter(f_out, fieldnames=line.keys(), delimiter="\t")
            # import pdb;pdb.set_trace()
            print(line)
            writer.writerow(line)

    # results to json
    # all_results = {"dev": results_devset, "test": results_testset}
    # if "-de" in test_dataset.get_filename():
    #     with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
    #         json.dump(all_results, f)
    # elif "-fr" in test_dataset.get_filename():
    #     with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
    #         json.dump(all_results, f)
    # else:
    #     logger.info(
    #         f"Was not able to deduct language from filename of testset, thus no metrics were saved.")
