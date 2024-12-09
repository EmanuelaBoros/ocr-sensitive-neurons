import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from peft import PeftModel

random_seed = 1  # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np

np.random.seed(random_seed)


def load_adapter_model(model_name, adapter_path):
    """
    Load the base model, its tokenizer, and the specified adapter.
    """
    # model_path = os.path.join(base_model_dir, model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # adapter_path = os.path.join(model_path, "adapters", adapter_name)

    # adapter_config = AdapterConfig.load(adapter_path)
    # model.add_adapter("test", AdapterType.text_task, config=adapter_config)
    # model.set_active_adapters("test")
    model = model.to("cuda")
    return model, tokenizer


# def generate_text(model, tokenizer, prompt):
#     """
#     Generate text using the provided model and tokenizer.
#     """
#     model_input = tokenizer(prompt, return_tensors="pt")
#     model.eval()
#
#     with torch.no_grad():
#         output = model.generate(**model_input, max_length=100)
#         generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     print(f"Generated text: {generated_text}")
#     print("-" * 50)
#     return generated_text


def generate_text(model, tokenizer, prompt):
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        answer = tokenizer.decode(
            model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True
        )
        return answer
    return None


def main():
    # adapter_path = "models/finetuned-corrected/mistralai/Mistral-7B-v0.1/checkpoints-corrected/checkpoint-1000/"
    adapter_dir = "mistral-hist-finetune/"
    adapter_path = "mistral-hist-finetune/checkpoint-25/"
    model_name = "mistralai/Mistral-7B-v0.1"

    print("Before adding the adapter", "*" * 150)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = ["Who is Leonardo Da Vinci?", "Qui est Léonard de Vinci ?", "Wer ist Leonardo Da Vinci?"]

    # Load model once and then load each checkpoint adapter into it
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model = base_model.to("cuda")

    print("Base model predictions", "*" * 150)
    for prompt in prompts:
        generated_text = generate_text(base_model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}\n")

    print("-" * 50)
    # Iterate over each checkpoint directory in the adapter directory
    for checkpoint in sorted(os.listdir(adapter_dir)):
        print(f"Adding the adapter from {checkpoint}", "*" * 150)
        checkpoint_path = os.path.join(adapter_dir, checkpoint)
        if os.path.isdir(checkpoint_path):
            print(f"After adding the adapter from {checkpoint_path}", "*" * 150)

            # Load the adapter from the checkpoint
            model_with_adapter = PeftModel.from_pretrained(base_model, checkpoint_path)
            model_with_adapter = model_with_adapter.to("cuda")

            for prompt in prompts:
                generated_text = generate_text(model_with_adapter, tokenizer, prompt)
                print(f"Prompt: {prompt}")
                print(f"Generated text: {generated_text}\n")


if __name__ == "__main__":
    main()
