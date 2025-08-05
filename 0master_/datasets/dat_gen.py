import os
import json
import glob
from random import shuffle
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

output_path = "Z:/dataset/andrew_data/todo_instruction_data.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2") 

task_queries = [
    "Add a task to my to-do list",
    "Show tasks due today",
    "Delete an item from my schedule",
    "Remind me about my deadline",
    "View all my tasks",
    "Set a reminder",
    "Add a calendar event"
]
task_embeddings = model.encode(task_queries, normalize_embeddings=True)

keywords = [
    "task", "to-do", "todo", "remind", "reminder", "due", "deadline",
    "schedule", "today", "tomorrow", "add", "remove", "delete",
    "list", "plan", "event", "appointment", "upcoming", "checklist"
]

filtered_data = []

def filter_oasst(dataset):
    count = 0
    for row in tqdm(dataset, desc="Processing OASST"):
        if row.get("role") != "assistant" or not row.get("text"):
            continue

        message_text = row["text"]
        msg_lower = message_text.lower()

        if any(k in msg_lower for k in keywords):
            sim = util.cos_sim(model.encode(message_text, normalize_embeddings=True), task_embeddings)
            if sim.max().item() > 0.4: 
                filtered_data.append({
                    "instruction": "[user prompt unavailable, context was filtered]",
                    "input": "",
                    "output": message_text.strip()
                })
                count += 1
    return count

def filter_by_keywords(dataset, prompt_key="instruction", response_key="output"):
    local_filtered = []
    for sample in dataset:
        prompt = sample.get(prompt_key, "").lower()
        response = sample.get(response_key, "").lower()
        combined = prompt + " " + response
        if any(kw in combined for kw in keywords):
            local_filtered.append({
                "instruction": sample.get(prompt_key, "").strip(),
                "input": "",
                "output": sample.get(response_key, "").strip()
            })
    return local_filtered

try:
    print("[+] Loading OpenAssistant...")
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    oasst_count = filter_oasst(oasst)
    print(f"    ✔ Semantically filtered {oasst_count} entries from OpenAssistant")
except Exception as e:
    print("OpenAssistant load failed:", e)

try:
    print("[+] Loading Dolly-15k...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_filtered = filter_by_keywords(dolly, prompt_key="instruction", response_key="response")
    filtered_data += dolly_filtered
    print(f"    ✔ Filtered {len(dolly_filtered)} entries from Dolly")
except Exception as e:
    print("Dolly load failed:", e)

try:
    print("[+] Loading Taskmaster (local)...")
    taskmaster_files = glob.glob("Z:/dataset/taskmaster_raw/TM-1-2019/data/*.json")
    tm_count = 0
    for file_path in taskmaster_files:
        with open(file_path, "r", encoding="utf-8") as f:
            dialogs = json.load(f)
            for dialog in dialogs:
                for utt in dialog.get("utterances", []):
                    text = utt.get("text", "")
                    if any(kw in text.lower() for kw in keywords):
                        filtered_data.append({
                            "instruction": text.strip(),
                            "input": "",
                            "output": "[Context-based response]"
                        })
                        tm_count += 1
    print(f"    ✔ Filtered {tm_count} entries from Taskmaster")
except Exception as e:
    print("Taskmaster load failed:", e)

print(f"[+] Total filtered examples before dedup: {len(filtered_data)}")

seen = set()
deduped_data = []
for item in filtered_data:
    key = (item["instruction"] + item["output"]).strip().lower()
    if key not in seen:
        seen.add(key)
        deduped_data.append(item)

shuffle(deduped_data)

with open(output_path, "w", encoding="utf-8") as f:
    for item in deduped_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[✅] Saved {len(deduped_data)} entries to: {output_path}")
