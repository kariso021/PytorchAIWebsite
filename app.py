from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
from model import load_model
from stable_diffusion import load_stable_diffusion_model, generate_image
import io
from PIL import Image

app = Flask(__name__)

# 모델 로드
model_path = 'model.pth'
model = load_model(model_path)
sd_model = load_stable_diffusion_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    if len(data) != 10:
        return jsonify({'error': 'Input data must have 10 elements'}), 400
    input_tensor = torch.tensor([data], dtype=torch.float32)  # 배치 차원 추가
    with torch.no_grad():
        output = model(input_tensor).item()
    return jsonify({'prediction': output})

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    image = generate_image(sd_model, prompt)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)