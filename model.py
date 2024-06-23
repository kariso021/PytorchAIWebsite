import torch
import torch.nn as nn
import torch.nn.functional as F

# 간단한 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 저장 함수
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 모델 로드 함수
def load_model(path):
    model = SimpleNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == "__main__":
    model = SimpleNN()
    save_model(model, 'model.pth')