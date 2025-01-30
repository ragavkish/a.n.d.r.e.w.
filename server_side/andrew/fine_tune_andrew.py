import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import gc
import os
import shutil
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_PATH = "Z:/kizX/dataset/andrew/models/anderson"
TEMP_PATH = "Z:/kizX/dataset/andrew/models/anderson_temp"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ds = load_dataset("OpenAssistant/oasst1")
train_ds = ds['train']
val_ds = ds['validation']

def format_conversations(example):
    messages = example["messages"]
    if len(messages) > 1:
        prompt = messages[0]["content"]
        response = messages[1]["content"]
        return {"prompt": prompt, "response": response}
    return {"prompt": messages[0]["content"], "response": ""}

formatted_train_ds = train_ds.map(format_conversations, remove_columns=["messages", "lang", "rank", "review_count", "model_name"])
formatted_val_ds = val_ds.map(format_conversations, remove_columns=["messages", "lang", "rank", "review_count", "model_name"])

def tokenize_data(example):
    text = f"{example['prompt']} {example['response']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train_ds = formatted_train_ds.map(tokenize_data, batched=True)
tokenized_val_ds = formatted_val_ds.map(tokenize_data, batched=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=TEMP_PATH,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    tokenizer=tokenizer
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
