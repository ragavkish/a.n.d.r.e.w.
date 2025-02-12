from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "Z:/kizX/dataset/andrew/models/mistral_finetuned"
CACHE_PATH = "Z:/kizX/dataset/andrew/cache"

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device_map = "auto" if device in ["cuda", "mps"] else None

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    cache_dir=CACHE_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    cache_dir=CACHE_PATH,
    low_cpu_mem_usage=True,  
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=device_map
)

model.eval()

def chat():
    print("A.N.D.R.E.W. is online! Type 'exit' to stop.")
    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("A.N.D.R.E.W.: Goodbye!")
            break

        chat_history.append(user_input)
        input_text = "\n".join(chat_history[-5:])

        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        input_ids = inputs["input_ids"].to(device, dtype=torch.float32)
        attention_mask = inputs["attention_mask"].to(device, dtype=torch.float32)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

        if output is None or output.shape[1] == 0:
            print("⚠️ Model did not generate any output! Retrying...")
            continue  

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        chat_history.append(response)
        print(f"A.N.D.R.E.W.: {response}")

if __name__ == "__main__":
    chat()
