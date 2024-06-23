from flask import Flask, request, jsonify, render_template, url_for
import torch
from torchvision import transforms
from model_cgan import Generator as CGANGenerator
from model_cyclegan import load_cyclegan_model, translate_image
from model import SimpleCNN, load_model, predict_image
from PIL import Image
import os
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# 모델 파일 경로
CNN_MODEL_PATH = 'cifar10_cnn.pth'
CGAN_MODEL_PATH = 'cgan_generator.pth'
CYCLEGAN_MODEL_PATH_A2B = 'G_A2B.pth'
CYCLEGAN_MODEL_PATH_B2A = 'G_B2A.pth'
UPLOAD_FOLDER = 'uploads'

# 업로드 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists('static'):
    os.makedirs('static')

# CNN 모델 로드
cnn_model = load_model(CNN_MODEL_PATH, SimpleCNN)

# CGAN 모델 로드 또는 생성
try:
    cgan_model = CGANGenerator()
    cgan_model.load_state_dict(torch.load(CGAN_MODEL_PATH))
    cgan_model.eval()
except FileNotFoundError:
    from train_cgan import train_cgan
    train_cgan(CGAN_MODEL_PATH)
    cgan_model = CGANGenerator()
    cgan_model.load_state_dict(torch.load(CGAN_MODEL_PATH))
    cgan_model.eval()

# CycleGAN 모델 로드 또는 생성
try:
    cyclegan_model_A2B = load_cyclegan_model(CYCLEGAN_MODEL_PATH_A2B)
    cyclegan_model_B2A = load_cyclegan_model(CYCLEGAN_MODEL_PATH_B2A)
except FileNotFoundError:
    from train_cyclegan import train_cyclegan
    train_cyclegan(CYCLEGAN_MODEL_PATH_A2B, CYCLEGAN_MODEL_PATH_B2A)
    cyclegan_model_A2B = load_cyclegan_model(CYCLEGAN_MODEL_PATH_A2B)
    cyclegan_model_B2A = load_cyclegan_model(CYCLEGAN_MODEL_PATH_B2A)

# Stable Diffusion 모델 로드
def load_stable_diffusion_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = pipe.to(device)
    return pipe

stable_diffusion_pipe = load_stable_diffusion_model()

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

@app.route('/cgan')
def cgan():
    return render_template('cgan.html')

@app.route('/generate-cgan', methods=['POST'])
def generate_cgan():
    label = request.json.get('label')
    if label is None:
        return jsonify({'error': '레이블이 제공되지 않았습니다'}), 400

    # CGAN 모델로 이미지 생성
    noise = torch.randn(1, 100)
    label = torch.tensor([label])
    image = cgan_model(noise, label).squeeze().cpu()

    # 이미지를 파일로 저장
    image_path = os.path.join('static', 'cgan_generated_image.png')
    transforms.ToPILImage()(image).save(image_path)
    
    # 이미지 URL 반환
    image_url = url_for('static', filename='cgan_generated_image.png')
    return jsonify({'image_url': image_url})

@app.route('/cyclegan')
def cyclegan():
    if not cyclegan_model_A2B or not cyclegan_model_B2A:
        return "CycleGAN model is not loaded. Please check the model file.", 500
    return render_template('cyclegan.html')

@app.route('/generate-cyclegan', methods=['POST'])
def generate_cyclegan():
    if not cyclegan_model_A2B or not cyclegan_model_B2A:
        return jsonify({'error': 'CycleGAN model is not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"File saved at: {file_path}")

        direction = request.form.get('direction', 'A2B')
        print(f"Direction: {direction}")
        if direction == 'A2B':
            model = cyclegan_model_A2B
        else:
            model = cyclegan_model_B2A

        image = translate_image(model, file_path)
        print(f"Output shape: {image.size}")

        # Ensure the static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')

        image_output_path = os.path.join('static', 'cyclegan_generated_image.png')
        image.save(image_output_path)
        print(f"Translated image saved at: {image_output_path}")

        image_url = url_for('static', filename='cyclegan_generated_image.png')
        print(f"Image URL: {image_url}")
        return jsonify({'image_url': image_url})

    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {str(e)}")  # 디버깅 출력
        return jsonify({'error': f'이미지 생성 중 오류 발생: {str(e)}'}), 500

@app.route('/stable-diffusion')
def stable_diffusion():
    return render_template('stable_diffusion.html')

@app.route('/generate-stable-diffusion', methods=['POST'])
def generate_stable_diffusion():
    print("Stable Diffusion 요청 수신됨")  # 디버깅 출력
    prompt = request.json.get('prompt')
    if prompt is None:
        return jsonify({'error': '프롬프트가 제공되지 않았습니다'}), 400

    # 프롬프트 길이 제한
    max_length = 100  # 예시로 100자로 제한
    if len(prompt) > max_length:
        return jsonify({'error': f'프롬프트는 {max_length}자를 초과할 수 없습니다'}), 400

    # Stable Diffusion 모델로 이미지 생성
    try:
        print("이미지 생성 시작")  # 디버깅 출력
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = stable_diffusion_pipe(prompt).images[0]
        print("이미지 생성 완료")  # 디버깅 출력
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {str(e)}")  # 디버깅 출력
        return jsonify({'error': f'이미지 생성 중 오류 발생: {str(e)}'}), 500

    try:
        image_path = os.path.join('static', 'stable_diffusion_generated_image.png')
        image.save(image_path)
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {str(e)}")  # 디버깅 출력
        return jsonify({'error': f'이미지 저장 중 오류 발생: {str(e)}'}), 500

    # 이미지 URL 반환
    image_url = url_for('static', filename='stable_diffusion_generated_image.png')
    return jsonify({'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
