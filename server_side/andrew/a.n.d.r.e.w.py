from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "Z:/kizX/dataset/andrew/models/anderson"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chat():
    print("A.N.D.R.E.W. is online! Type 'exit' to stop.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("A.N.D.R.E.W.: Goodbye!")
            break

        chat_history.append(user_input)
        input_text = " ".join(chat_history[-5:])

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + 50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )

        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        chat_history.append(response)
        print(f"A.N.D.R.E.W.: {response}")

if __name__ == "__main__":
    chat()
