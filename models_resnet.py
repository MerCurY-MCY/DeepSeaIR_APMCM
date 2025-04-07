import torch.nn as nn
import torch
'''
残差块的核心思想是跳跃连接，是让网络中的某些层的输入直接跳跃到后面的某些层，而不经过中间的卷积、激活等操作。
传统的卷积神经网络（CNN）通常是将每一层的输入通过一系列卷积层、激活函数等处理，最终得到输出。
但随着网络深度增加，容易发生 梯度消失 或 梯度爆炸，使得训练变得非常困难。
残差块的出现，通过直接将输入传递到后面的层，从而使得训练过程变得更稳定。
简而言之，残差块不仅学习输入到输出的直接映射，还学习输入和输出之间的残差（residual），也就是两者的差值。

假设 
𝑥是输入，经过若干层的卷积操作之后得到的输出是 𝐹(𝑥)
传统网络会直接将 x映射到输出 𝐹(𝑥)，而在残差网络中，我们学习的是残差
（即 𝐹(𝑥)和 𝑥之间的差值），然后再加上输入 𝑥，得到最终的输出：
Output=𝐹(𝑥) + 𝑥
这里的 𝐹(𝑥)是通过卷积层学习到的映射，而 𝑥是输入数据，Output就是残差块的输出。

为什么残差块有效？
1) 缓解梯度消失问题：
传统的深度网络在层数增加时，训练中可能出现 梯度消失（vanishing gradient） 问题，即误差在反向传播时会变得非常小，导致更新无法有效进行。残差块通过直接传递输入，使得梯度可以直接传递到更深层次，从而避免了梯度消失的问题。
2) 更容易优化：
对于深层网络，残差块让网络学习的任务变得更加容易——它学习的是输入和输出之间的 残差，而不是直接学习映射。这使得优化更加容易，因为学习残差比学习直接映射要简单。
换句话说，网络只需要学习如何改变输入，而不是从零开始学会映射所有特征。
3) 更高效的训练：
在没有残差块时，深度网络往往会陷入局部最小值，导致训练困难。而引入残差块后，网络学习的目标变得更加明确，从而可以获得更好的训练效果。
4) 加速收敛：
残差连接使得反向传播的梯度信号能够更容易地通过较深的层传播，这有助于加速训练收敛，尤其是在深度网络中。

在残差网络中中间层的地位：
在传统的网络中，数据从输入通过每一层的操作（如卷积、池化等）逐步传递，直到最后一层得到输出。但是，这样的结构在深层网络中容易导致 梯度消失 或 梯度爆炸，训练变得困难，甚至在一些情况下，深层网络的性能还不如浅层网络。
为了缓解这个问题，ResNet 中引入了 跳跃连接，使得输入 𝑥可以绕过中间的卷积层，直接与最终输出相加。这种结构的核心思想是学习残差，而不是直接学习输入到输出的映射。
在引入跳跃连接时，并不是直接跳过中间层，而是通过以下几种方式来保留中间层的作用：
1.每一个残差块内部的层（如卷积层、激活函数、批归一化等）依然起到重要的作用。
它们负责提取和转换输入特征，只是网络的【学习目标】转变为学习输入和输出之间的差异，而不是学习从输入到输出的完整映射。
2.跳跃连接只是加和操作，将中间层的输出与输入直接相加。因此，尽管输入通过中间的层进行处理，但这些层的输出与输入之间的残差相加，有助于避免梯度消失，且网络能够更容易地调整这些中间层的权重。
3.实际上，中间层的操作依然非常重要，跳跃连接只是在输出中加入了原始输入的信息，以帮助网络更有效地训练。通过这种方式，网络在学习过程中能够 直接传递信息，并且调整特征提取的方式，使得每一层都能学习到不同层次的特征。

总结：
虽然跳跃连接通过直接传递输入来加速训练，但这并不意味着中间层被忽略了。事实上，残差块内的每一层（如卷积层和激活函数）都对特征学习起着至关重要的作用。
残差块的设计并没有跳过这些层，而是让网络可以通过残差学习的方式，使得每一层的输出能够有效地与输入信息结合，从而加快网络的收敛并提升性能。

为什么通过把网络有学习目标变成学习输出与输入之间的差值，再加上输出中加入原始数据，就可以缓解那些问题？
在训练深层神经网络时，我们通常会遇到以下几个问题：

a. 梯度消失/梯度爆炸
随着网络层数增加，反向传播时，梯度的传播可能会变得非常小（梯度消失）或者非常大（梯度爆炸），使得训练变得困难。尤其是在深层网络中，误差梯度在每一层都会被反向传播，而梯度在传递过程中逐渐缩小或者放大，导致优化无法有效进行。

b. 学习目标变得复杂
在传统的神经网络中，模型要直接学习从输入到输出的映射。如果网络足够深，学习的目标变得非常复杂，且容易陷入局部最优，无法有效地训练网络。

c. 训练过程中的不稳定性
随着层数的增加，网络可能会变得非常难以训练，尤其是在不良初始化或选择不合适的激活函数的情况下。网络的训练可能非常慢，甚至出现收敛性问题。

为什么残差（差值）学习有效？
a. 简化学习目标
在传统神经网络中，我们要求网络 学习从输入到输出的直接映射。随着网络层数的增加，这个映射变得非常复杂。

而在 残差网络 (ResNet) 中，网络的目标从学习输入到输出的映射，转变为学习输入和输出之间的 差值（即残差）。因此，我们要求网络学习的目标变得更加简单和明确。

为什么残差学习更简单？
a. 简化学习目标
假设你的目标函数是从输入 映射到输出 即 y=f(x)。
如果网络很深，学习这样的映射可能非常困难。相反，网络只需要学习如何将输入与目标输出之间的差异（残差）加起来：
这样，网络只需要学习“差距”，而不是完全从输入到输出的映射。这让网络变得更容易训练，因为它实际上是在 学习微小的调整，而不是重新学习从头开始的全局映射。
b. 避免梯度消失和梯度爆炸
残差块通过 跳跃连接，使得输入 
𝑥直接传递到更深的层，并与网络的输出相加。这种结构有几个好处：
梯度更容易传播：在反向传播时，跳跃连接为梯度提供了直接路径，避免了梯度逐渐消失。即使网络非常深，梯度仍然能够有效传播到网络的较浅层，保持了梯度的 稳定性。因此，网络的训练变得更加稳定，避免了梯度消失或爆炸的问题。
更好的梯度流动：通过跳跃连接，残差网络允许梯度直接流向前面的层，并且由于网络学习的是差异（残差），梯度可以在网络中流动得更加高效。这样，梯度可以快速传递给较浅层的网络，帮助它们有效地更新权重。
c. 加速训练过程
由于网络的学习目标变得更加简化，网络不再需要学习非常复杂的映射。残差块 降低了训练的难度，使得训练更容易且更快。
另外，残差块还 减少了信息的损失。在传统的网络中，输入信息会被每一层逐渐变换和压缩，可能会丢失一些有用的信息。而残差块通过跳跃连接让输入信息能够与输出结合，从而有效保留输入的 关键信息，避免过多的信息丢失。

为什么加上输入信息（跳跃连接）可以缓解这些问题？
a. 信息保留
通过直接将输入 添加到输出 F(x) 中（即进行加法操作），我们保留了输入的原始信息，并且它与网络学习到的特征相结合，形成了最终的输出。
这种方式有助于网络更好地保留输入的原始特征，避免在深度网络中丢失关键信息。
b. 增强网络的表达能力
即使网络非常深，跳跃连接允许网络学习残差而不是整个函数映射。这样，每一层网络都能基于前面的信息调整输出，而不需要从零开始。因此，网络更容易学到复杂的特征，并且更容易进行优化。
c. 避免退化问题
在非常深的网络中，随着层数增加，模型的表现可能会下降，这被称为 退化问题。残差网络通过引入跳跃连接，实际上在某种程度上避免了这一问题。由于每个残差块都可以学习到一小步的映射，即使网络层数增加，模型的性能也不会退化。

跳跃连接：跳跃连接的意思是，在一个残差块的内部，输入和经过残差块卷积层处理后的输出连接，而不是残差块和残差块的连接。


常见的残差块结构
通常，一个标准的残差块包括：
第一个卷积层：通常是一个 
1.3×3 的卷积核，步长通常为 1，填充（padding）为 1，这样保持输入输出的大小一致。
ReLU激活函数：对卷积结果应用 ReLU 激活函数。
2.第二个卷积层：与第一个卷积层相似，通常也是 3×3 的卷积核，步长为 1，填充为 1。
3.跳跃连接：输入信号（或者某些情况下经过1x1卷积调整后的信号）与第二个卷积层的输出相加，得到残差块的最终输出。
总结起来，一个典型的残差块有 两个卷积层。但是有些更复杂的残差块可以包含更多的卷积层、批归一化（BatchNorm）、不同尺寸的卷积核、甚至包含不同的处理方式。

'''
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
        # 由于在定义1x1卷积层时，设定了out_channels。这意味着每个输入像素点（每个位置的所有通道特征）都会通过out_channels个不同的卷积核进行加权求和，最终生成128个输出通道
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


# Generator with residual blocks
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 退化图像处理分支
        self.degraded_image_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
        )

        # PSF处理分支（相差特征处理）
        self.psf_branch = nn.Sequential(
            nn.Conv1d(256 * 19, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            ResidualBlock(512, 256),  # 使用残差块处理PSF特征
            nn.ReLU(),
        )

        # 融合
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 44 * 44 + 256, 1024)

        # 图像恢复分支
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, degraded_image, psf_matrix):
        x = self.degraded_image_branch(degraded_image)
        x = self.flatten(x)

        psf_features = self.psf_branch(psf_matrix)
        combined = torch.cat([x, psf_features], dim=1)

        x = self.fc1(combined)
        x = x.view(-1, 512, 1, 1)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        output_image = self.deconv4(x)

        return output_image
# Discriminator with residual blocks
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 退化图像处理分支
        self.degraded_image_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
        )

        # 融合退化图像特征
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 44 * 44, 1024)  # 扁平化后的特征输入全连接层
        self.fc2 = nn.Linear(1024, 1)  # 输出真假判断（0 或 1）

    def forward(self, degraded_image):
        # 退化图像分支
        x = self.degraded_image_branch(degraded_image)
        x = self.flatten(x)  # 展平特征

        # 判别器最终判断
        x = self.fc1(x)
        x = torch.leaky_relu(x, negative_slope=0.2)
        output = self.fc2(x)  # 输出真假判断（0 或 1）

        return output


