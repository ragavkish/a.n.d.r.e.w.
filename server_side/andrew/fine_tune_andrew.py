import os
import torch
import gc
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers.utils import logging

logging.set_verbosity_info()
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

CACHE_PATH = "Z:/kizX/dataset/andrew/cache"
MODEL_PATH = "Z:/kizX/dataset/andrew/models/mistral_finetuned"
TEMP_PATH = "Z:/kizX/dataset/andrew/models/mistral_temp"

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LOCAL_MODEL_PATH = f"{CACHE_PATH}/models--mistralai--Mistral-7B-v0.1"
USE_LOCAL_MODEL = os.path.exists(LOCAL_MODEL_PATH)
model_name = LOCAL_MODEL_PATH if USE_LOCAL_MODEL else BASE_MODEL
print(f"Loading model from: {model_name}")

os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_PATH, use_fast=False)
except Exception as e:
    print(f"Tokenizer error: {e}. Re-downloading...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_PATH, force_download=True)

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_PATH, torch_dtype=torch.float16)
except Exception as e:
    print(f"Model error: {e}. Re-downloading...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=CACHE_PATH, torch_dtype=torch.float16, force_download=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

DATASET_NAME = "OpenAssistant/oasst1"
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_PATH)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    save_total_limit=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("test", None),
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(MODEL_PATH)
print("Standard fine-tuning completed and model saved!")

print("Loading OpenAssistant dataset for LoRA fine-tuning...")
ds = load_dataset(DATASET_NAME, cache_dir=CACHE_PATH)
train_ds, val_ds = ds["train"], ds.get("validation", None)

def extract_conversations(dataset):
    conversations = []
    last_user_message = None
    for example in dataset:
        if example["role"].lower() in ["prompter", "user"]:
            last_user_message = example["text"]
        elif example["role"].lower() == "assistant" and last_user_message:
            conversations.append({"prompt": last_user_message, "response": example["text"]})
            last_user_message = None
    return Dataset.from_list(conversations) if conversations else None

formatted_train_ds = extract_conversations(train_ds)
formatted_val_ds = extract_conversations(val_ds) if val_ds else None

def tokenize_data(examples):
    texts = [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(texts, truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_train_ds = formatted_train_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])
tokenized_val_ds = formatted_val_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"]) if formatted_val_ds else None

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
)

model = get_peft_model(model, lora_config)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    logits, labels = torch.tensor(logits, device=device), torch.tensor(labels, device=device)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    return {"eval_loss": loss_fn(logits, labels).item()}

lora_training_args = TrainingArguments(
    output_dir=TEMP_PATH,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)

lora_trainer = Trainer(
    model=model,
    args=lora_training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    compute_metrics=compute_metrics
)

lora_trainer.train()

del lora_trainer
torch.cuda.empty_cache()
gc.collect()
model.cpu()

if os.path.exists(MODEL_PATH):
    shutil.move(MODEL_PATH, f"{MODEL_PATH}_backup")

shutil.move(TEMP_PATH, MODEL_PATH)
print(f"LoRA fine-tuned Mistral-7B model saved at: {MODEL_PATH}")
