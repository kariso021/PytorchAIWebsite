import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline

# GAN 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28)

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 저장 함수
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 모델 로드 함수
def load_model(path, model_class):
    model = model_class()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("모델이 성공적으로 로드되었습니다.")
    else:
        print(f"모델 파일을 찾을 수 없습니다. 새 모델을 생성하고 저장합니다: {path}")
        save_model(model, path)
    model.eval()
    return model

# Stable Diffusion 모델 로드 함수
def load_stable_diffusion_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    return pipe

# GAN 이미지 생성 함수
def generate_gan_image(model, noise_dim=100):
    noise = torch.randn(1, noise_dim)
    with torch.no_grad():
        generated_image = model(noise).squeeze().cpu()
    return transforms.ToPILImage()(generated_image)

# 예측 함수
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()