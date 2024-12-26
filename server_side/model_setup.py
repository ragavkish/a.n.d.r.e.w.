from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "models\\anderson"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()
