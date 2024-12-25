from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

@app.route('/')
def home():
    return render_template('a.n.d.r.e.w.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('prompt', '')

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': response_text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
