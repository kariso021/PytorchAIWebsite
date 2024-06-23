import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model_cyclegan import Generator, Discriminator
from itertools import cycle
from PIL import Image
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, transform=None, data_dir='./data/trainA'):
        self.transform = transform
        self.data_dir = data_dir
        self.data = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def train_cyclegan(generator_path_A2B, generator_path_B2A):
    # Hyperparameters
    batch_size = 1
    learning_rate = 0.0002
    num_epochs = 5000  # Increase the number of epochs for sufficient training
    lambda_cycle = 10.0
    lambda_identity = 0.5 * lambda_cycle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Actual datasets
    dataset_A = ImageDataset(transform=transform, data_dir='./data/trainA')
    dataset_B = ImageDataset(transform=transform, data_dir='./data/trainB')
    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    # Initialize models
    G_A2B = Generator(input_nc=3, output_nc=3).to(device)
    G_B2A = Generator(input_nc=3, output_nc=3).to(device)
    D_A = Discriminator(input_nc=3).to(device)
    D_B = Discriminator(input_nc=3).to(device)

    # Losses
    adversarial_loss = nn.MSELoss().to(device)
    cycle_loss = nn.L1Loss().to(device)
    identity_loss = nn.L1Loss().to(device)

    # Optimizers
    g_optimizer = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    d_A_optimizer = optim.Adam(D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_B_optimizer = optim.Adam(D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Training
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(cycle(loader_A), cycle(loader_B))):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            batch_size = real_A.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size, *D_A.output_shape).to(device)
            fake = torch.zeros(batch_size, *D_A.output_shape).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            G_A2B.train()
            G_B2A.train()

            # Identity loss
            loss_id_A = identity_loss(G_B2A(real_A), real_A)
            loss_id_B = identity_loss(G_A2B(real_B), real_B)

            # GAN loss
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = adversarial_loss(D_B(fake_B), valid)
            fake_A = G_B2A(real_B)
            loss_GAN_B2A = adversarial_loss(D_A(fake_A), valid)

            # Cycle loss
            recov_A = G_B2A(fake_B)
            loss_cycle_A = cycle_loss(recov_A, real_A)
            recov_B = G_A2B(fake_A)
            loss_cycle_B = cycle_loss(recov_B, real_B)

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + lambda_cycle * (loss_cycle_A + loss_cycle_B) + lambda_identity * (loss_id_A + loss_id_B)

            g_optimizer.zero_grad()
            loss_G.backward()
            g_optimizer.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            fake_A = G_B2A(real_B).detach()
            loss_real_A = adversarial_loss(D_A(real_A), valid)
            loss_fake_A = adversarial_loss(D_A(fake_A), fake)
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5

            d_A_optimizer.zero_grad()
            loss_D_A.backward()
            d_A_optimizer.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            fake_B = G_A2B(real_A).detach()
            loss_real_B = adversarial_loss(D_B(real_B), valid)
            loss_fake_B = adversarial_loss(D_B(fake_B), fake)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5

            d_B_optimizer.zero_grad()
            loss_D_B.backward()
            d_B_optimizer.step()

            if i % 200 == 0:  # 매 200번째 배치마다 로그를 출력
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/10] [D loss: {loss_D_A.item() + loss_D_B.item()}] [G loss: {loss_G.item()}]")

        # Save the models periodically
        if epoch % 100 == 0:
            torch.save(G_A2B.state_dict(), f"{generator_path_A2B}_epoch_{epoch}.pth")
            torch.save(G_B2A.state_dict(), f"{generator_path_B2A}_epoch_{epoch}.pth")

    # Save the final models
    torch.save(G_A2B.state_dict(), generator_path_A2B)
    torch.save(G_B2A.state_dict(), generator_path_B2A)
    print(f'CycleGAN models saved to {generator_path_A2B} and {generator_path_B2A}')

if __name__ == '__main__':
    train_cyclegan('G_A2B.pth', 'G_B2A.pth')
Key Enhancements: