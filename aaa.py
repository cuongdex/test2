import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
import os

# Đường dẫn đến dataset trên local
data_dir = "/content/drive/MyDrive/PetImages"

# Chuyển đổi và chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize(16),            # Thay đổi kích thước ảnh về 32x32
    transforms.CenterCrop(16),        # Cắt giữa ảnh để đảm bảo đúng kích thước
    transforms.ToTensor(),            # Chuyển đổi ảnh thành tensor
    transforms.Normalize([0.5], [0.5]) # Chuẩn hóa ảnh về khoảng [-1, 1]
])

# Tạo dataset từ local
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Các bước tiếp theo sử dụng train_loader như đã mô tả trước đây
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalUNet2DModel(UNet2DModel):
    def forward(self, x, timesteps, class_labels):
        # Convert class_labels to one-hot encoding
        class_embeddings = nn.functional.one_hot(class_labels, num_classes=2).float().to(x.device)
        class_embeddings = class_embeddings.unsqueeze(-1).unsqueeze(-1)
        class_embeddings = class_embeddings.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate class_embeddings with x
        x = torch.cat([x, class_embeddings], dim=1)
        return super().forward(x, timesteps)

# Sử dụng model này để huấn luyện thay vì UNet2DModel ban đầu
model = ConditionalUNet2DModel(
    sample_size=16,
    in_channels=3 + 2,  # 3 channels for image, 2 for class labels
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


