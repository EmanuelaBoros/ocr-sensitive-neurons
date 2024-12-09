import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer


# Define the function to load the model and tokenizer
def load_model_and_tokenizer(base_model, quant_config, guanaco_dataset):
    dataset = load_dataset(guanaco_dataset, split="train")
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, dataset


# Define the function for training
def train_model(model, tokenizer, dataset, peft_params, training_params):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    return trainer


# Function to start TensorBoard
def start_tensorboard(log_dir):
    from tensorboard import notebook

    notebook.start("--logdir {} --port 4000".format(log_dir))


# Function to generate text
def generate_text(pipe, prompt):
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]["generated_text"]


def train(model, tokenizer, dataset, new_model):

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()
    # Save trained model and tokenizer
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)


if __name__ == "__main__":
    base_model = "NousResearch/Llama-2-7b-chat-hf"
    guanaco_dataset = "mlabonne/guanaco-llama2-1k"
    new_model = "llama-2-7b-chat-guanaco"
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model, tokenizer, dataset = load_model_and_tokenizer(base_model, quant_config, guanaco_dataset)

    print("Before training")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

    prompt = "Who is Leonardo Da Vinci?"
    print(generate_text(pipe, prompt))

    prompt = "What is Datacamp Career track?"
    print(generate_text(pipe, prompt))

    train(model, tokenizer, dataset, new_model)

    print("-" * 30)
    print("After training")
    # log_dir = "results/runs"
    # start_tensorboard(log_dir)

    # Set logging verbosity
    logging.set_verbosity(logging.CRITICAL)

    # model = AutoModelForCausalLM.from_pretrained(new_model, quantization_config=quant_config, device_map={"": 0})

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

    prompt = "Who is Leonardo Da Vinci?"
    print(generate_text(pipe, prompt))

    prompt = "What is Datacamp Career track?"
    print(generate_text(pipe, prompt))
