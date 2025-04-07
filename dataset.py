import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from PIL import Image
import os
from torchvision import transforms
import random
# 数据集定义
class SpaceTelescopeDataset(Dataset):
    def __init__(self, degraded_img_dir, real_img_dir, psf_dir, transform=None):
        """
        初始化数据集类
        :param degraded_img_dir: 退化图像的存储路径
        :param real_img_dir: 真实图像的存储路径
        :param psf_dir: 相差矩阵存储路径
        :param transform: 图像预处理转换
        """
        self.degraded_img_dir = degraded_img_dir
        self.real_img_dir = real_img_dir
        self.psf_dir = psf_dir
        self.transform = transform

        # 获取退化图像和真实图像的文件名列表
        self.degraded_imgs = sorted(os.listdir(degraded_img_dir))
        self.real_imgs = sorted(os.listdir(real_img_dir))
        self.psf_files = sorted(os.listdir(psf_dir))
        indices = list(range(len(self.degraded_imgs)))
        random.shuffle(indices)  # 打乱索引顺序
        self.degraded_imgs = [self.degraded_imgs[i] for i in indices]
        self.psf_files = [self.psf_files[i] for i in indices]
    def __len__(self):
        # 返回数据集的大小
        return len(self.degraded_imgs)

    def __getitem__(self, idx):
        """
        获取每一个样本，包括退化图像，真实图像和PSF矩阵
        :param idx: 索引
        :return: 一个包含退化图像、真实图像和PSF矩阵的字典
        """
        degraded_img_path = os.path.join(self.degraded_img_dir, self.degraded_imgs[idx])
        degraded_img = Image.open(degraded_img_path).convert('L')  # 灰度图
        psf_path = os.path.join(self.psf_dir, self.psf_files[idx])
        psf_data = sio.loadmat(psf_path)
        psf_matrix = psf_data['map']  # 假设矩阵名为'PSF'
        psf_matrix = np.expand_dims(psf_matrix, axis=0)  # 将PSF矩阵扩展为(1, 256, 19)

        real_img_path = os.path.join(self.real_img_dir, self.real_imgs[idx % len(self.real_imgs)])
        real_img = Image.open(real_img_path).convert('L')






        # 将图像转换为Tensor
        if self.transform:
            degraded_img = self.transform(degraded_img)
            real_img = self.transform(real_img)

        # 转换PSF矩阵为Tensor
        psf_matrix = torch.tensor(psf_matrix, dtype=torch.float32)
        print("Degraded Images:", self.degraded_imgs)
        print("Real Images:", self.real_imgs)
        print("PSF Files:", self.psf_files)
        return {'degraded_img': degraded_img, 'real_img': real_img, 'psf_matrix': psf_matrix}

# 图像预处理
'''
在的残差块和生成器以及判别器中，已经有批归一化操作。批归一化会在每层的前向传播中标准化数据，使其均值为0，方差为1，帮助网络训练更加稳定，并且能够在训练时动态适应数据分布。
这意味着，网络已经在内部处理了数据标准化或归一化的操作。

批归一化的作用是在每层进行标准化，但它并不等同于图像输入的归一化。通常，图像数据会在输入网络前进行归一化，以便让网络的输入数据分布更加均匀。最常见的做法是将图像像素值从 [0, 255] 的范围缩放到 [0, 1] 或 [-1, 1] 之间，这对于很多神经网络来说是一种标准的做法。
因此，依然可能需要在数据输入时进行归一化，而不仅仅依赖于网络中的批归一化。


所以，尽管网络中已经有了批归一化层，这并不意味着你不需要对输入数据进行归一化。在处理图像数据时，通常我们会将图像像素值从 [0, 255] 的范围归一化到 [0, 1] 或 [-1, 1]，以提高训练的稳定性和效率。

为什么仍然需要归一化：
数据范围统一：归一化操作帮助确保数据的范围适合神经网络的输入要求，尤其是在使用 ReLU 激活函数时，过大的输入值可能导致梯度消失或爆炸。
更快的收敛：归一化后的数据有助于加速模型的训练过程，因为它使得网络能够更快地找到最优解。
避免数值溢出：像素值在[0, 255]时，可能会导致某些激活函数（如Sigmoid）或计算中出现数值溢出，归一化可以避免这一问题。

归一化的通常作用：
1. 加速训练和提高收敛速度
神经网络的训练过程通常依赖于梯度下降算法。如果输入数据的尺度差异很大（例如，一些特征的数值范围在[0, 1]之间，其他特征在[1000, 10000]之间），
模型在训练时会遇到困难，因为较大的特征值可能导致梯度计算不平衡，从而使得网络更新的步长不一致，导致学习过程不稳定。
归一化将所有输入数据映射到相同的尺度（通常是[0, 1]或[-1, 1]），使得所有特征对训练过程的影响更加平衡，从而加速了收敛速度。
2. 防止数值溢出和梯度爆炸/消失
神经网络中的激活函数（如Sigmoid, Tanh等）在数值范围过大或过小的输入上可能会导致梯度爆炸或梯度消失问题。
比如，Sigmoid函数的输出范围是[0, 1]，当输入非常大时，梯度接近零，导致权重更新缓慢，甚至停止学习（梯度消失）。如果输入的数值范围太大，也可能导致梯度爆炸，使得训练过程不稳定。
通过归一化输入数据，可以使得输入数据的值落在一个合理的范围内，避免这些问题的发生。
3. 确保输入数据的均匀分布
神经网络往往假设输入特征在训练时服从某种分布，归一化操作使得输入数据在模型训练时分布更加均匀。
例如，通常使用零均值和单位方差的分布来规范化数据，这样可以让每个特征对模型的贡献是平等的，从而提高模型的效果。
如果数据分布不均匀，某些特征可能会占主导地位，影响模型的学习过程和结果。归一化确保每个特征具有相同的尺度，避免数据中的某些特征对模型训练产生过大或过小的影响。
4. 适应优化器
优化算法（如梯度下降、Adam等）通常依赖于输入数据的标准化。如果输入数据没有归一化，可能会导致优化器在更新过程中采取不一致的步长。
例如，某些特征值大、某些小的情况下，优化器可能会忽略小的特征，从而降低了优化效率。
归一化可以使得优化器在所有维度上以相同的步长进行更新，有助于更快、更准确地找到最优解。
5. 提升模型性能
虽然归一化并不是直接影响模型结构的因素，但它可以提升模型的训练效果和性能。
在没有适当归一化的情况下，可能需要更多的训练步骤、更多的正则化或更复杂的调整才能达到满意的效果。
归一化通常能减少模型调参的复杂性，使得模型更容易学习到有效的特征。
6. 有助于特定模型架构的稳定性
在某些类型的神经网络中，特别是深度神经网络（DNNs）和卷积神经网络（CNNs）中，归一化有助于提高网络的稳定性。
例如，在卷积神经网络（CNNs）中，输入图像的归一化确保了图像的像素值都处于适合网络学习的范围，避免了激活函数和卷积计算中可能的数值问题。
'''
transform = transforms.Compose([
    transforms.Resize((581, 581)),  # 确保图像大小为581x581
    transforms.ToTensor(),  # 转换为Tensor，归一化到[0, 1]
])

# # 创建数据集
# degraded_img_dir = 'C:/Users/Li Mu/Desktop/GPU图像退化/D_data_1208'
# real_img_dir = 'C:/Users/Li Mu/Desktop/GPU图像退化/data'
# psf_dir = 'C:/Users/Li Mu/Desktop/GPU图像退化/mat'
#
# dataset = SpaceTelescopeDataset(degraded_img_dir, real_img_dir, psf_dir, transform=transform)
#
# # 数据加载器
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 测试加载数据

