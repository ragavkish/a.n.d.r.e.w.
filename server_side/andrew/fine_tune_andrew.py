import torch
import numpy as np
import evaluate
import gc
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

MODEL_PATH = "Z:/kizX/dataset/andrew/models/anderson"
TEMP_PATH = "Z:/kizX/dataset/andrew/models/anderson_temp"

torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ds = load_dataset("OpenAssistant/oasst1")
train_ds = ds["train"]
val_ds = ds["validation"]

all_roles = set([example["role"] for example in train_ds])
user_role = next((r for r in all_roles if "prompt" in r.lower()), "prompter")
assistant_role = next((r for r in all_roles if "assistant" in r.lower()), "assistant")

def extract_conversations(dataset):
    conversations = []
    last_user_message = None

    for example in dataset:
        if example["role"] == user_role:
            last_user_message = example["text"]
        elif example["role"] == assistant_role and last_user_message:
            conversations.append({"prompt": last_user_message, "response": example["text"]})
            last_user_message = None

    return Dataset.from_list(conversations) if conversations else None

formatted_train_ds = extract_conversations(train_ds)
formatted_val_ds = extract_conversations(val_ds)

if not formatted_train_ds or not formatted_val_ds:
    raise ValueError("Extracted dataset is empty. Check role mapping and dataset structure.")

def tokenize_data(examples):
    texts = [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_train_ds = formatted_train_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])
tokenized_val_ds = formatted_val_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])

if len(tokenized_train_ds) == 0 or len(tokenized_val_ds) == 0:
    raise ValueError("Tokenized dataset is empty. Check the extraction and tokenization steps.")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["transformer.h.0.attn.c_attn", "transformer.h.0.attn.c_proj"],
)

model = get_peft_model(model, lora_config)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    logits = torch.tensor(np.array(logits)).detach().cpu()
    labels = torch.tensor(np.array(labels)).detach().cpu()

    loss_fn = torch.nn.CrossEntropyLoss()
    
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    loss = loss_fn(logits, labels).item()
    
    return {"eval_loss": loss}

training_args = TrainingArguments(
    output_dir=TEMP_PATH,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=1,
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    prediction_loss_only=True
)

trainer.train()

del trainer
torch.cuda.empty_cache()
gc.collect()
model.cpu()

if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
shutil.move(TEMP_PATH, MODEL_PATH)

print(f"Fine-tuned model saved to {MODEL_PATH}")
