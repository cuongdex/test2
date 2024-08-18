import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler

# Chuyển đổi và chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize(32),            # Thay đổi kích thước ảnh về 16x16
    transforms.CenterCrop(32),        # Cắt giữa ảnh để đảm bảo đúng kích thước
    transforms.ToTensor(),            # Chuyển đổi ảnh thành tensor
    transforms.Normalize([0.5], [0.5]) # Chuẩn hóa ảnh về khoảng [-1, 1]
])

# Tạo dataset
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# Các bước tiếp theo sử dụng train_loader như đã mô tả trước đây
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalUNet2DModel(UNet2DModel):
    def forward(self, x, timesteps, class_labels):
        # Convert class_labels to one-hot encoding
        class_embeddings = nn.functional.one_hot(class_labels, num_classes=10).float().to(x.device)
        class_embeddings = class_embeddings.unsqueeze(-1).unsqueeze(-1)
        class_embeddings = class_embeddings.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate class_embeddings with x
        x = torch.cat([x, class_embeddings], dim=1)
        return super().forward(x, timesteps)

# Sử dụng model này để huấn luyện thay vì UNet2DModel ban đầu
model = ConditionalUNet2DModel(
    sample_size=32,
    in_channels=3 + 10,  # 3 channels for image, 2 for class labels
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "AttnUpBlock2D"),
).to(device)

# Lịch trình nhiễu DDPM
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Cấu hình bộ tối ưu hóa
optimizer = Adam(model.parameters(), lr=1e-4)


# Hàm huấn luyện (như trong mã trước)
def train_model(model, train_loader, noise_scheduler, optimizer, num_epochs=10, device='cpu'):
    model.train()  # Đặt mô hình vào chế độ huấn luyện

    for epoch in range(num_epochs):
        for step, (images, class_labels) in enumerate(train_loader):
            # Đưa ảnh và nhãn lớp vào thiết bị (GPU hoặc CPU)
            images = images.to(device)
            class_labels = class_labels.to(device)

            # Tạo nhiễu ngẫu nhiên và thêm vào ảnh
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Đặt gradient về 0 trước khi thực hiện backward pass
            optimizer.zero_grad()

            # Gọi mô hình với noisy_images, timesteps và class_labels
            pred_noise = model(noisy_images, timesteps, class_labels).sample

            # Tính toán tổn thất giữa nhiễu dự đoán và nhiễu thực bằng MSE Loss
            loss = nn.MSELoss()(pred_noise, noise)

            # Thực hiện backward pass để tính toán gradient
            loss.backward()

            # Cập nhật các tham số của mô hình dựa trên gradient
            optimizer.step()

            # In ra thông tin về tiến trình huấn luyện
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

# Gọi hàm này để bắt đầu huấn luyện mô hình
train_model(model, train_loader, noise_scheduler, optimizer, num_epochs=1,device='cpu')