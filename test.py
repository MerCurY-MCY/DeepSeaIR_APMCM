import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models import Generator

# 自定义数据集类，仅加载单路径图像
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.tiff', '.jpeg'))  # 支持常见图片格式
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

# 配置路径
test_dir = r".\test1"  # 测试集路径
checkpoint_dir = "checkpoints"  # 模型参数保存路径
output_dir = "test_results2"  # 生成结果保存路径
os.makedirs(output_dir, exist_ok=True)

# 数据变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 加载测试数据集
test_dataset = TestDataset(test_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载生成器模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G_AB = Generator().to(device)  # 退化 -> 清晰
G_AB.load_state_dict(torch.load(f"{checkpoint_dir}/G_AB_epoch_50.pth"))  # 加载训练完成的模型参数
G_AB.eval()  # 切换到评估模式

# 测试阶段
print("开始测试...")
with torch.no_grad():  # 禁用梯度计算
    for i, (degraded, img_path) in enumerate(test_loader):
        degraded = degraded.to(device)

        # 生成增强图像
        enhanced = G_AB(degraded)

        # 保存结果
        img_name = os.path.basename(img_path[0])  # 获取原始文件名
        save_image(enhanced, f"{output_dir}/enhanced_{img_name}")
        save_image(degraded, f"{output_dir}/degraded_{img_name}")

        print(f"已处理 {i+1}/{len(test_loader)} 张图像：{img_name}")

print(f"测试完成！结果保存在 {output_dir}")
