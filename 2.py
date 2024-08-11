from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Tải mô hình pretrained Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
unet = pipeline.unet
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer

# Đặt mô hình ở chế độ training
unet.train()
text_encoder.train()

# Định nghĩa optimizer và learning rate
optimizer = torch.optim.AdamW([
    {"params": unet.parameters(), "lr": 5e-5},
    {"params": text_encoder.parameters(), "lr": 5e-5},
])

# Định nghĩa lịch trình diffusion
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Training loop
for epoch in range(5):  # Tăng số epoch nếu cần
    for batch in train_dataloader:
        # Chuẩn bị dữ liệu
        images = batch['image'].to(device)
        texts = batch['text']

        # Encode văn bản
        input_ids = tokenizer(texts, return_tensors="pt", padding=True).input_ids.to(device)

        # Tạo nhiễu cho hình ảnh
        noise = torch.randn_like(images).to(device)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps=torch.randint(0, 1000, (images.size(0),)).to(device))

        # Forward pass
        encoded_texts = text_encoder(input_ids)[0]
        pred_noise = unet(noisy_images, encoded_texts)["sample"]

        # Tính loss
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        # Backward pass và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


pipeline.save_pretrained("my-stable-diffusion-cats-and-dogs")


prompt = "A cute dog playing with a cat"
image = pipeline(prompt).images[0]
image.save("dog_and_cat.png")
