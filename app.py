from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from model import GPTLanguageModel

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
model.to(device)

itos = {...}
stoi = {...}

def decode(sequence):
    return ''.join([itos[i] for i in sequence])

def encode(sequence):
    return [stoi[c] for c in sequence]

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('prompt', '')

    # Encode the input text
    input_ids = torch.tensor([encode(input_text)], dtype=torch.long, device=device)

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100)
    response_text = decode(output[0].tolist())

    return jsonify({'response': response_text.strip()})

if __name__ == '__main__':
    app.run(debug=True)