import cv2
import numpy as np
import os
from glob import glob

# 参数设置
omega = 0.95  # 雾量系数，控制透射率的保留比例
t_min = 0.1   # 最小透射率，防止除零问题
patch_size = 15  # 暗通道计算的窗口大小
output_folder = "./enhanced_images222"  # 保存增强图片的文件夹

# 创建保存增强图片的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 计算暗通道
def calculate_dark_channel(image, patch_size):
    """
    计算图像的暗通道
    Args:
        image: 输入图像 (BGR 格式)
        patch_size: 窗口大小
    Returns:
        dark_channel: 暗通道图像
    """
    min_channel = np.min(image, axis=2)  # 每个像素取最小的颜色通道值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)  # 最小值滤波
    return dark_channel

# 估计背景光
def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    """
    估计背景光
    Args:
        image: 输入图像 (BGR 格式)
        dark_channel: 暗通道图像
        top_percent: 用于估计背景光的亮点百分比
    Returns:
        atmospheric_light: 背景光
    """
    h, w = dark_channel.shape
    num_pixels = int(h * w * top_percent)
    dark_vec = dark_channel.ravel()
    indices = np.argsort(dark_vec)[-num_pixels:]  # 选择暗通道值最高的像素

    brightest = image.reshape(-1, 3)[indices]
    atmospheric_light = np.mean(brightest, axis=0)
    return atmospheric_light

# 计算透射率
def calculate_transmission(image, atmospheric_light, omega, patch_size):
    """
    计算透射率
    Args:
        image: 输入图像 (BGR 格式)
        atmospheric_light: 背景光
        omega: 雾量系数
        patch_size: 窗口大小
    Returns:
        transmission: 透射率图像
    """
    norm_image = image / atmospheric_light
    dark_channel = calculate_dark_channel(norm_image, patch_size)
    transmission = 1 - omega * dark_channel
    return transmission

# 恢复图像
def recover_image(image, transmission, atmospheric_light, t_min):
    """
    恢复清晰图像
    Args:
        image: 输入图像 (BGR 格式)
        transmission: 透射率图像
        atmospheric_light: 背景光
        t_min: 最小透射率
    Returns:
        recovered_image: 恢复后的图像
    """
    # 确保透射率在 [t_min, 1] 范围内
    transmission = np.clip(transmission, t_min, 1)

    # 将透射率扩展到与图像匹配的三通道
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

    # 恢复图像
    recovered_image = (image - atmospheric_light) / transmission + atmospheric_light
    return np.clip(recovered_image, 0, 255).astype(np.uint8)


# 处理所有图片
input_images = glob('./test/*.jpg')  # 假设所有图片在 './input_images/' 文件夹下
for img_path in input_images:
    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"Cannot read {img_path}, skipping...")
        continue

    # 计算暗通道
    dark_channel = calculate_dark_channel(image, patch_size)

    # 估计背景光
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)

    # 计算透射率
    transmission = calculate_transmission(image, atmospheric_light, omega, patch_size)

    # 恢复图像
    enhanced_image = recover_image(image, transmission, atmospheric_light, t_min)

    # 保存结果
    file_name = os.path.basename(img_path)
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, enhanced_image)
    print(f"Processed and saved: {output_path}")

print("All images have been processed and saved!")
