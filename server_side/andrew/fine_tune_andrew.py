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

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_PATH)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_PATH, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("your_dataset", cache_dir=CACHE_PATH)

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
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("Z:/kizX/dataset/andrew/fine_tuned_model")
print("Standard fine-tuning completed and model saved!")

ds = load_dataset("OpenAssistant/oasst1", cache_dir=CACHE_PATH)
train_ds, val_ds = ds["train"], ds["validation"]

all_roles = set(example["role"] for example in train_ds)
user_role = next((r for r in all_roles if "prompter" in r.lower() or "user" in r.lower()), None)
assistant_role = next((r for r in all_roles if "assistant" in r.lower()), None)

if not user_role or not assistant_role:
    raise ValueError("Role mapping failed. Check dataset format!")

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
    raise ValueError("Extracted dataset is empty! Check dataset structure.")

def tokenize_data(examples):
    texts = [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(texts, truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_train_ds = formatted_train_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])
tokenized_val_ds = formatted_val_ds.map(tokenize_data, batched=True, remove_columns=["prompt", "response"])

if len(tokenized_train_ds) == 0 or len(tokenized_val_ds) == 0:
    raise ValueError("Tokenized dataset is empty! Check extraction & tokenization steps.")

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
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits, device=device)
    labels = torch.tensor(labels, device=device)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    loss = loss_fn(logits, labels).item()
    return {"eval_loss": loss}

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
    backup_path = f"{MODEL_PATH}_backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.move(MODEL_PATH, backup_path)

shutil.move(TEMP_PATH, MODEL_PATH)
print(f"LoRA fine-tuned Mistral-7B model saved at: {MODEL_PATH}")
