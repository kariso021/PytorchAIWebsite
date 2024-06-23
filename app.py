from flask import Flask, request, jsonify, render_template, send_file, url_for
import torch
import numpy as np
from model import SimpleCNN, Generator, load_model, load_stable_diffusion_model, generate_gan_image, predict_image
from PIL import Image
import io
import os

app = Flask(__name__)

# 모델 파일 경로
CNN_MODEL_PATH = 'cifar10_cnn.pth'
GAN_MODEL_PATH = 'generator.pth'
UPLOAD_FOLDER = 'uploads'

# 업로드 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# CNN 모델 로드
cnn_model = load_model(CNN_MODEL_PATH, SimpleCNN)

# Stable Diffusion 모델 로드
stable_diffusion_pipe = load_stable_diffusion_model()

# GAN 모델 로드
gan_model = load_model(GAN_MODEL_PATH, Generator)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-classification')
def image_classification():
    return render_template('image_classification.html')

@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 이미지 예측
        class_id = predict_image(file_path, cnn_model)
        class_name = classes[class_id]
        return jsonify({'class_id': class_id, 'class_name': class_name})

@app.route('/stable-diffusion')
def stable_diffusion():
    return render_template('stable_diffusion.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt')
    if prompt is None:
        return jsonify({'error': '프롬프트가 제공되지 않았습니다'}), 400

    # Stable Diffusion 모델로 이미지 생성
    with torch.cuda.amp.autocast():
        image = stable_diffusion_pipe(prompt).images[0]
    
    # 이미지를 파일로 저장
    image_path = os.path.join('static', 'generated_image.png')
    image.save(image_path)
    
    # 이미지 URL 반환
    image_url = url_for('static', filename='generated_image.png')
    return jsonify({'image_url': image_url})

@app.route('/gan')
def gan():
    return render_template('gan.html')

@app.route('/generate-gan', methods=['POST'])
def generate_gan():
    # GAN 모델로 이미지 생성
    image = generate_gan_image(gan_model)
    
    # 이미지를 파일로 저장
    image_path = os.path.join('static', 'gan_generated_image.png')
    image.save(image_path)
    
    # 이미지 URL 반환
    image_url = url_for('static', filename='gan_generated_image.png')
    return jsonify({'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)