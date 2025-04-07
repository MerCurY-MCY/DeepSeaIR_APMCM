import torch
import torch.nn as nn
from pytorch_msssim import ssim

# 生成器损失（包括对抗损失和SSIM损失）
def generator_loss(fake_preds, real_images, fake_images, lambda_ssim=1.0):
    adversarial_loss = nn.MSELoss()(fake_preds, torch.ones_like(fake_preds))  # 对抗损失
    ssim_loss = 1 - ssim(fake_images, real_images, data_range=1.0)  # SSIM损失
    return adversarial_loss + lambda_ssim * ssim_loss, adversarial_loss, ssim_loss

# 判别器损失
def discriminator_loss(real_preds, fake_preds):
    real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))  # 真实图像损失
    fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))  # 假图像损失
    return real_loss + fake_loss

# 循环一致性损失
def cycle_consistency_loss(real_image, recovered_image, lambda_cycle=10.0):
    # 使用L1损失来计算循环一致性损失
    return lambda_cycle * nn.L1Loss()(recovered_image, real_image)
