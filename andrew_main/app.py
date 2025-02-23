import os
import random
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "Z:/kizX/dataset/andrew/models"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def greet():
    print("Hello! I'm ANDREW..")
    name = input("Enter your name: ")
    return name

def chatbot():
    print("Starting chat with ANDREW...")
    name = greet()
    
    while True:
        user_input = input(f"{name}: ")
        if user_input.lower() in ["quit", "exit", "leave"]:
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"ANDREW: {response}")

if __name__ == "__main__":
    chatbot()
