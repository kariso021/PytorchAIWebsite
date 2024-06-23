import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Generator 모델 정의
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

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, 784))

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# 모델, 손실 함수 및 옵티마이저 정의
netG = Generator()
netD = Discriminator()
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

# 모델 학습
num_epochs = 50
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(trainloader):
        # 진짜 데이터
        real = data
        label = torch.ones(data.size(0), 1)
        
        # 가짜 데이터
        noise = torch.randn(data.size(0), 100)
        fake = netG(noise)
        label_fake = torch.zeros(data.size(0), 1)
        
        # Discriminator 학습
        optimizerD.zero_grad()
        output = netD(real)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        output = netD(fake.detach())
        lossD_fake = criterion(output, label_fake)
        lossD_fake.backward()
        optimizerD.step()
        
        # Generator 학습
        optimizerG.zero_grad()
        output = netD(fake)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{num_epochs}] Loss D: {lossD_real+lossD_fake}, Loss G: {lossG}')

    if epoch % 10 == 0:
        save_image(fake.data[:25], f'output_{epoch}.png', nrow=5, normalize=True)

# 모델 저장
torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')