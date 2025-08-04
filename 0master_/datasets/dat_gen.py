import os
import json
from datasets import load_dataset
from random import shuffle

output_path = "Z:/dataset/andrew_data/todo_instruction_data.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

keywords = [
    "task", "to-do", "todo", "remind", "reminder", "due", "deadline",
    "schedule", "today", "delete task", "view list", "add task", "upcoming task"
]

filtered_data = []

def filter_by_keywords(dataset, prompt_key="prompt", response_key="response"):
    local_filtered = []
    for sample in dataset:
        prompt = sample.get(prompt_key, "").lower()
        response = sample.get(response_key, "").lower()
        if any(kw in prompt for kw in keywords):
            local_filtered.append({
                "instruction": sample.get(prompt_key, ""),
                "input": "",
                "output": sample.get(response_key, "")
            })
    return local_filtered


try:
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    filtered_data += filter_by_keywords(oasst, prompt_key="prompt", response_key="response")
except Exception as e:
    print("OpenAssistant load failed:", e)

try:
    tm1 = load_dataset("taskmaster", "taskmaster1", split="train")
    for dialog in tm1:
        for message in dialog.get("utterances", []):
            text = message.get("text", "")
            if any(kw in text.lower() for kw in keywords):
                filtered_data.append({
                    "instruction": text,
                    "input": "",
                    "output": "[Response depends on prior context]"
                })
except Exception as e:
    print("Taskmaster1 load failed:", e)

try:
    flan = load_dataset("tiangolo/flan-ul2", split="train[:2%]") 
    filtered_data += filter_by_keywords(flan, prompt_key="inputs", response_key="targets")
except Exception as e:
    print("FLAN UL2 load failed:", e)

print(f"Total filtered entries: {len(filtered_data)}")
shuffle(filtered_data)

seen = set()
deduped_data = []
for entry in filtered_data:
    key = entry["instruction"].strip().lower()
    if key not in seen:
        seen.add(key)
        deduped_data.append(entry)

with open(output_path, "w", encoding="utf-8") as f:
    for item in deduped_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(deduped_data)} entries to {output_path}")
