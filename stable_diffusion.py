import torch
from pytorch_stable_diffusion import StableDiffusion

# Stable Diffusion 모델 로드 함수
def load_stable_diffusion_model():
    model = StableDiffusion.load_model("CompVis/stable-diffusion-v1-4", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# 이미지 생성 함수
def generate_image(model, prompt):
    images = model.generate(prompt, num_inference_steps=50, guidance_scale=7.5)
    image = images[0]
    return image