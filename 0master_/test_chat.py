import json
from pathlib import Path
from llama_cpp import Llama

def load_file(path):
    return Path(path).read_text(encoding='utf-8') if Path(path).exists() else ""

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_memory(path="memory.json"):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except:
        return {}

def build_system_prompt(mode="default"):
    persona = load_file("0master_/priming/persona.txt")
    memory = load_memory()
    few_shot = load_jsonl("0master_/priming/few_shot.jsonl")


    memory_text = "\n".join(f"- {k}: {v}" for k, v in memory.items())
    mode_text = load_file(f"modes/{mode}.txt") if mode != "default" else ""

    system_prompt = f"""{persona.strip()}

CURRENT MEMORY:
{memory_text or 'No memory available'}

MODE SETTINGS:
{mode_text.strip()}
"""

    return system_prompt, few_shot

llm = Llama(
    model_path="Z:/models/andrew_modelz/mistral-7b-openorca.Q5_K_M.gguf",
    chat_format="mistral-instruct",
    n_ctx=4096,
    n_threads=8,
    verbose=False,
    seed=42
)

mode = "default"
system_prompt, few_shot = build_system_prompt(mode)

chat_history = [{"role": "system", "content": system_prompt}]
for example in few_shot:
    chat_history.extend(example["messages"])

print("A.N.D.R.E.W. is online.\nType 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        print("A.N.D.R.E.W.: Shutting down. Take care, Kish.")
        break

    chat_history.append({"role": "user", "content": user_input})

    response = llm.create_chat_completion(
        messages=chat_history,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["</s>"]
    )

    reply = response['choices'][0]['message']['content'].strip()
    print("A.N.D.R.E.W.:", reply)

    chat_history.append({"role": "assistant", "content": reply})
