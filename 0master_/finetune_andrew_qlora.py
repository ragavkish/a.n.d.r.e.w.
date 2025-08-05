import os
import subprocess
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel

HF_MODEL = "Open-Orca/Mistral-7B-OpenOrca"
DATASET_PATH = r"Z:/dataset/andrew_data/todo_instruction_data.jsonl"
OUTPUT_BASE = r"Z:/models/andrew_modelz"
LORA_OUTPUT = os.path.join(OUTPUT_BASE, "andrew_mistral_lora")
MERGED_MODEL_DIR = os.path.join(OUTPUT_BASE, "andrew_mistral_full")
FINAL_GGUF = os.path.join(OUTPUT_BASE, "andrew_mistral.Q5_K_M.gguf")

BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LEN = 512
MAX_STEPS = 2500
LR = 2e-4
EPOCHS = 2

if not os.path.exists(os.path.join(OUTPUT_BASE, "Mistral-7B-OpenOrca")):
    print("[+] Downloading full precision Mistral-7B-OpenOrca model...")
    subprocess.run([
        "git", "clone", "https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca",
        os.path.join(OUTPUT_BASE, "Mistral-7B-OpenOrca")
    ])
    subprocess.run(["git", "lfs", "pull"], cwd=os.path.join(OUTPUT_BASE, "Mistral-7B-OpenOrca"))
else:
    print("[✔] Full precision model already exists locally.")

model_name = os.path.join(OUTPUT_BASE, "Mistral-7B-OpenOrca")

print("[+] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16
    )
    device_map = "auto"
else:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )
    device_map = {"": "cpu"}

print("[+] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=quant_config
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("[+] Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH)
dataset = dataset.shuffle(seed=42)

def format_example(example):
    if example["input"].strip():
        prompt = f"""Below is an instruction and an input. Respond appropriately.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""
    else:
        prompt = f"""Below is an instruction. Respond appropriately.

### Instruction:
{example['instruction']}

### Response:
"""
    return {
        "input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN).input_ids,
        "labels": tokenizer(example["output"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN).input_ids
    }

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

print("[+] Starting fine-tuning...")
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=100,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
    num_train_epochs=EPOCHS,
    fp16=False,
    bf16=False,
    resume_from_checkpoint=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

trainer.train()

print("[+] Merging LoRA weights into base model...")
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
lora_model = PeftModel.from_pretrained(base_model, LORA_OUTPUT)
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

llama_cpp_path = "path/to/llama.cpp"
if os.path.exists(os.path.join(llama_cpp_path, "convert.py")):
    print("[+] Converting to GGUF Q5_K_M format...")
    subprocess.run([
        "python", "convert.py",
        MERGED_MODEL_DIR,
        "--outfile", FINAL_GGUF,
        "--model", "mistral"
    ], cwd=llama_cpp_path)

    subprocess.run([
        "./quantize", FINAL_GGUF, FINAL_GGUF, "Q5_K_M"
    ], cwd=llama_cpp_path)

print(f"[✅] Fine-tuning complete. New model saved at: {FINAL_GGUF}")
print("Replace your old .gguf with this one and run test_chat.py to see improved A.N.D.R.E.W.")
