import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cgan import Generator, Discriminator

def train_cgan(generator_path):
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.0002
    num_epochs = 5000
    latent_size = 100
    num_classes = 10

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            outputs = discriminator(images, labels)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            noise = torch.randn(batch_size, latent_size).to(device)
            fake_images = generator(noise, labels)
            outputs = discriminator(fake_images, labels)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            noise = torch.randn(batch_size, latent_size).to(device)
            fake_images = generator(noise, labels)
            outputs = discriminator(fake_images, labels)

            g_loss = criterion(outputs, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # Save the generator model
    torch.save(generator.state_dict(), generator_path)
    print(f'cGAN model saved to {generator_path}')

if __name__ == '__main__':
    train_cgan('cgan_generator.pth')