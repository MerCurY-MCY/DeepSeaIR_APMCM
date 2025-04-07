import torch
import torch.nn as nn

import torch.nn.functional as F
'''
unet架构：
编码器的主要任务是从输入图像中 提取特征，通过一系列的卷积层和池化层，编码器将输入图像中的低级特征（如边缘、纹理、形状等）提取出来，并逐步加深这些特征的抽象级别。
随着编码器的深入，图像的空间分辨率会减少，但特征的深度和抽象程度会增加。
解码器的主要任务是通过一系列的反卷积（转置卷积）操作将编码器输出的低分辨率、高语义的特征图 转换回 高分辨率、像素级的分割图。解码器的目标是恢复图像的空间细节，以便进行像素级的分割。

总之编码器的任务是提取特征，通过卷积和池化逐步抽象和压缩输入图像的特征，得到一个低分辨率的特征图，包含了图像的语义信息。
解码器的任务是 恢复空间信息，通过反卷积将编码器生成的低分辨率特征图恢复成高分辨率的输出图像，并借助跳跃连接恢复更多细节，最终输出像素级的分割结果。


unet中的跳跃连接：
在 U-Net 中，跳跃连接的主要作用是 将编码器（下采样部分）和解码器（上采样部分）之间的特征信息直接连接起来。
这种连接通常是在编码器中的每一层和解码器对应的上采样层之间建立的，即直接将编码器某一层的输出传递给解码器中的相应层，以帮助解码器恢复更多的细节。
跳跃连接的主要优势如下：
1.保留高分辨率的特征：编码器中的下采样过程通常会导致图像空间分辨率的降低（通过卷积和池化等操作）。这种操作会使得图像的细节信息丢失，而这些细节信息对于分割任务非常重要。跳跃连接能够将这些细节信息直接传递到解码器，避免在解码过程中丢失重要的空间信息。
2.加强低级特征和高级特征的结合：通过跳跃连接，解码器可以同时利用编码器提取的低级特征（例如边缘、纹理等）和高级语义特征（例如对象的形状、位置等）。这使得解码器不仅能恢复图像的空间结构，还能增强模型的细节分割能力。
3.提高模型的性能：跳跃连接能够帮助模型更好地恢复图像细节，减少因网络过深或过度下采样导致的信息丢失，通常会显著提高模型在像素级任务（如图像分割）上的性能。
在 U-Net 中，跳跃连接的操作通常是将编码器和解码器对应层的输出特征图进行拼接（concatenate），而拼接后的特征图会传递给解码器的下一层。这时，拼接后的特征图会被进一步处理，通常包括卷积、激活和批归一化等操作
跳跃连接正是让解码器在恢复图像的过程中，同时利用编码器提取的低级特征和高级语义特征。
具体来说：
低级特征：在编码器的早期层，网络主要学习图像中的简单、细节性的特征，比如 边缘、纹理、颜色变化 等。通过跳跃连接，解码器能够直接获取这些低级特征，帮助它在恢复图像的过程中保持细节。
高级特征：随着网络深入，编码器的后续层会提取更加抽象和高级的特征，如对象的形状、位置、结构信息 等。跳跃连接确保解码器不仅能看到低级特征，还能获取这些高级特征，帮助它生成更符合原始图像语义的复原结果。
'''
# U-Net编码器部分

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        # 第一层卷积 + 批归一化 + 激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二层卷积 + 批归一化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #nn.Sequential() 是一个容器，用于按顺序执行多个层
        #这里是为了了给残差连接提供一个占位符。但如果它没有传入任何层，那么它就表示一个空的操作，意味着该部分不会对输入进行任何变换。
        #而如果输入图的通道和残差块的输出通道不匹配，就需要下面的卷积层进行处理
        self.shortcut = nn.Sequential()
        # 残差连接：输入和输出的通道数不匹配时，使用1x1卷积，in_channels：输入通道数，out_channels：输出通道数
        # 1x1 卷积 的主要目的是通过调整通道数来改变特征图的深度，而不改变图像的空间尺寸，卷积核的尺寸是 1x1，它只覆盖输入的每个像素点，不进行空间上的扩展。
        # 由于在定义1x1卷积层时，设定了out_channels。这意味着每个输入像素点（每个位置的所有通道特征）都会通过out_channels个不同的卷积核进行加权求和，最终生成out_channels个输出通道
        # 所以最终，图像的宽度和高度保持不变，而通道数改变
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 卷积层1 -> 激活函数 -> 批归一化
        out = self.relu(self.bn1(self.conv1(x)))

        # 卷积层2 -> 批归一化
        out = self.bn2(self.conv2(out))

        # 加上残差连接
        out += self.shortcut(x)

        # 最后通过ReLU激活函数
        out = self.relu(out)

        return out


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels1=128, out_channels2=256):
        super(UNetEncoder, self).__init__()
        # 第一层卷积：输入通道为 in_channels，输出通道为 128
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.relu = nn.ReLU(inplace=True)

        # 第二层卷积：输入通道为 128，输出通道为 256
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels2)

    def forward(self, x):
        # 第一层卷积和激活
        x1 = self.relu(self.bn1(self.conv1(x)))  # 第一层卷积
        # 第二层卷积和激活
        x2 = self.relu(self.bn2(self.conv2(x1)))  # 第二层卷积
        # 返回两个特征图用于跳跃连接
        return x1, x2


# U-Net解码器部分
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()

        # 使用 ConvTranspose2d 进行上采样
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


        self.conv_adjust_channels = nn.Conv2d(384,out_channels, kernel_size=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)

    def forward(self, x, skip_connection):
        # 第一层反卷积
        x = self.relu(self.bn1(self.deconv1(x)))


        # 调整跳跃连接的尺寸并拼接
        skip_connection = F.interpolate(skip_connection, size=(x.size(2), x.size(3)), mode='bilinear',
                                        align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_adjust_channels(x)

        # 第二层反卷积
        x = self.relu(self.bn1(self.deconv2(x)))
        return x

# Generator 使用 U-Net
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        # 退化图像处理分支（编码器）
        self.degraded_image_encoder = UNetEncoder(1, 256)  # 输入1个通道的图像，输出64个通道的特征

        # PSF处理分支（相差特征处理）
        #结合unet和resnet的特点

        # 图像恢复解码器
        self.deconv1 = UNetDecoder(256, 128)
        self.deconv2 = UNetDecoder(128, 64)

        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, degraded_image, psf_matrix):
        # 退化图像编码分支
        x1, x2 = self.degraded_image_encoder(degraded_image)  # 生成跳跃连接特征

        # PSF特征处理
        # psf_features = self.psf_branch(psf_matrix)
        #
        # # 合并PSF特征与退化图像特征
        # x = torch.cat([x2, psf_features], dim=1)  # 在通道维度拼接
        x = x2

        # 解码器阶段
        x = self.deconv1(x, x1)  # 第一层解码并融合跳跃连接
        x = self.deconv2(x, x2)
        output_image = self.deconv3(x)  # 输出图像

        return output_image


# Discriminator 使用 U-Net（假设输入为退化图像）
class DiscriminatorUNet(nn.Module):
    def __init__(self):
        super(DiscriminatorUNet, self).__init__()

        # 退化图像处理分支（编码器）
        self.degraded_image_encoder = UNetEncoder(1, 64)

        # 池化层减少特征图的尺寸
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化（GAP）
        self.fc1 = nn.Linear(64, 1024)  # 使用池化后的32个通道
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, degraded_image):
        # 退化图像编码分支
        x1, x2 = self.degraded_image_encoder(degraded_image)

        # 使用全局平均池化，减少维度
        x2 = self.global_pool(x2)  # 输出形状: (batch_size, 32, 1, 1)
        x2 = torch.flatten(x2, 1)  # 展平为 (batch_size, 32)

        # 全连接层
        x = self.fc1(x2)
        x = F.leaky_relu(x, negative_slope=0.2)
        output = self.fc2(x)

        return output
