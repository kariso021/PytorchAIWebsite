import torch
import torch.optim as optim
from model import SimpleNN, save_model  # 모델과 저장 함수 가져오기

# 모델 인스턴스화
model = SimpleNN()

# 임의의 데이터 생성
x_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# 손실 함수 및 옵티마이저 정의
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 모델 학습
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 모델 저장
save_model(model, 'model.pth')
print("모델이 성공적으로 저장되었습니다.")