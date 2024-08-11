from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Tải dataset từ Hugging Face hoặc từ thư mục local
data_dir = "Petlmages"

# Định nghĩa các phép biến đổi cho dữ liệu
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Tạo dataset từ thư mục local
dataset = ImageFolder(root=data_dir, transform=transform)

# Tạo DataLoader để huấn luyện mô hình
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)




