from flask import Blueprint, request, jsonify, render_template
from server_side.model_setup import model, tokenizer, device
import torch

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template('a.n.d.r.e.w.html')

@routes.route('/generate', methods=['POST'])
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

def init_routes(app):
    app.register_blueprint(routes)
