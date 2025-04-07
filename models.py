import torch.nn as nn
import torch


import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 退化图像处理分支
        self.degraded_image_branch = nn.Sequential(
            # 第1层卷积：输入1通道图像，输出64通道，步长2下采样
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 第2层卷积：输出128通道，步长2下采样
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # 第3层卷积：输出256通道，步长2下采样
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # 第4层卷积：输出512通道，步长2下采样
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        # 添加残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(512) for _ in range(6)]  # 6个残差块
        )

        # 图像恢复分支（反卷积），卷积的逆过程，恢复图像
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 输出层
        self.sigmoid = nn.Sigmoid() # 使用Tanh激活函数，输出在[-1, 1]范围内

    def forward(self, x):
        # 退化图像处理分支
        x = self.degraded_image_branch(x)

        # 通过残差块
        x = self.residual_blocks(x)

        # 图像恢复（上采样）
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        # 最终输出
        x = self.sigmoid(x)

        return x


'''
对抗训练的交替过程：
第一步:首先训练判别器D来判断真实图像和生成图像的差异。此时，生成器 𝐺被固定，判别器通过学习判断生成图像是否真实（即输出接近 1 还是 0）。
（在该任务中，退化图像和生成图像将被输入到判别器。判别器 D负责判断哪些图像是真实的，哪些是生成器G生成的
在没有成对的真实图像时，判别器的任务依然是学习区分退化图像和恢复图像的差异，判断生成的图像是否能够逼近真实的图像。
在这种情况下，判别器 𝐷的目标是判断图像是否为“真实”的恢复图像，或者是由生成器 G生成的。
在这个过程中，生成器暂时不参与更新，只是用于生成图像供判别器进行训练。
生成器会用退化图像和像差特征作为输入，生成恢复图像（尽管这些恢复图像与真实图像相比还不够好）。
判别器就会基于这些图像进行训练，逐渐学会区分真实与生成图像。
）
第二步：训练生成器𝐺。此时，判别器D被固定，生成器试图生成能“欺骗”判别器的图像。生成器通过优化对抗损失，让判别器无法区分它生成的图像与真实图像的区别。

第三步：交替进行：这两个步骤交替进行，生成器不断学习如何生成更真实的图像，而判别器则不断提高判断能力。
'''
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.4, inplace=True) )
            layers.append(nn.Dropout(0.5))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),

            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, recover ):
        # Concatenate image and condition image by channels to produce input
        img_input = recover
        return self.model(img_input)


