import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image

# 간단한 신경망 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
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
    else:
        print(f"Model file not found. Creating a new model and saving it to {path}")
        save_model(model, path)
    model.eval()
    return model

# 이미지 예측 함수
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()