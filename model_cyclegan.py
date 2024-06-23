import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            # 추가적인 레이어들
            # ...
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),  # 출력 채널을 3으로 맞춤
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

def load_cyclegan_model(model_path):
    model = Generator()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    return model

def translate_image(model, image_path):
    import cv2
    import numpy as np
    from PIL import Image

    # Load the image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    # Convert image to the format expected by the model (e.g., normalize, resize)
    image = cv2.resize(image, (256, 256))  # Adjust the size if necessary
    image = image / 127.5 - 1  # Normalize to [-1, 1]

    # Convert image to PyTorch tensor
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Run the model
    with torch.no_grad():
        output = model(image).squeeze(0).permute(1, 2, 0).numpy()

    # Debugging output
    print(f"Output shape: {output.shape}")

    # Check the number of channels
    if output.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels: {output.shape[2]}")

    # Convert the output back to an image
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype('uint8')
    output_image = Image.fromarray(output)

    return output_image