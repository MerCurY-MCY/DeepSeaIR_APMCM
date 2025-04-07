import os
import numpy as np
import random
import imageio
import cv2
from tqdm import tqdm


# 应用退化函数
def apply_degradation(image, degradation_type):
    """
    根据退化类型对图像进行退化处理
    :param image: 输入图像
    :param degradation_type: 退化类型（"blue_shift", "green_shift", "low_light", "blur"）
    :return: 退化后的图像
    """
    N_lambda = {
        "blue_shift": [0.2, 0.4, 0.9],  # 蓝偏
        "green_shift": [0.3, 0.9, 0.5],  # 绿偏
        "low_light": [0.5, 0.5, 0.5],  # 低光
    }

    if degradation_type in ["blue_shift", "green_shift", "low_light"]:
        # 随机生成深度图
        depth = np.random.uniform(0.5, 15) * generate_fake_depth(image.shape)

        # 计算传输率 T_x
        T_c = N_lambda[degradation_type]
        T_x = np.ndarray((image.shape[0], image.shape[1], 3))
        for i in range(3):
            T_x[:, :, i] = T_c[i] * depth
        T_x = (T_x - T_x.min()) / (T_x.max() - T_x.min())

        # 背景光 B_lambda
        B_lambda = np.ndarray((image.shape[0], image.shape[1], 3))
        for i in range(3):
            B_lambda[:, :, i].fill(np.random.uniform(1.2, 1.8) * T_c[i])

        # 生成退化图像
        degraded_image = image * T_x + B_lambda * (1 - T_x)
        degraded_image = (degraded_image - degraded_image.min()) / (degraded_image.max() - degraded_image.min())
        return degraded_image

    if degradation_type == "blur":
        # 随机选择模糊核大小
        blur_level = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (blur_level, blur_level), 0)

    # 如果无退化，返回原始图像
    return image


# 生成假深度图
def generate_fake_depth(image_shape):
    """
    生成模拟深度图
    :param image_shape: 图像形状
    :return: 深度图
    """
    height, width = image_shape[:2]
    xv, yv = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    depth = 1 - np.sqrt(xv ** 2 + yv ** 2)  # 中心为最近，边缘为最远
    return depth


# 主函数
def process_and_rename_images(input_dirs, renamed_base_dir, degraded_base_dir):
    """
    读取原始分类图像，重命名并保存，同时生成退化图像
    :param input_dirs: 分类文件夹字典 {类别: 路径}
    :param renamed_base_dir: 重命名后的图像存储基础路径
    :param degraded_base_dir: 退化图像存储基础路径
    """
    os.makedirs(renamed_base_dir, exist_ok=True)
    os.makedirs(degraded_base_dir, exist_ok=True)

    for category, input_dir in input_dirs.items():
        # 创建对应的分类文件夹
        renamed_dir = os.path.join(renamed_base_dir, category)
        degraded_dir = os.path.join(degraded_base_dir, category)
        os.makedirs(renamed_dir, exist_ok=True)
        os.makedirs(degraded_dir, exist_ok=True)

        # 遍历当前分类的原始图像
        for idx, img_file in tqdm(enumerate(sorted(os.listdir(input_dir))), desc=f"Processing {category}"):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 读取并归一化原始图像
            img_path = os.path.join(input_dir, img_file)
            org_img = imageio.imread(img_path) / 255.0
            if len(org_img.shape) == 2:  # 灰度图转 RGB
                org_img = np.stack([org_img] * 3, axis=-1)

            # 保存重命名后的原始图像
            renamed_file = os.path.join(renamed_dir, f"{idx}.png")
            imageio.imwrite(renamed_file, (org_img * 255).astype(np.uint8))

            # 生成并保存退化图像
            for i in range(6):
                degradation_type = category  # 使用当前分类作为退化类型
                degraded_img = apply_degradation(org_img, degradation_type)
                degraded_file = os.path.join(degraded_dir, f"{idx}_{i}.png")
                imageio.imwrite(degraded_file, (degraded_img * 255).astype(np.uint8))


# 主程序入口
if __name__ == "__main__":
    input_dirs = {
        "blue_shift": "C:/Users/Li Mu/Desktop/classified/color_distortion_blue",
        "green_shift": "C:/Users/Li Mu/Desktop/classified/color_distortion_green",
        "low_light": "C:/Users/Li Mu/Desktop/classified/low_light",
        "blur": "C:/Users/Li Mu/Desktop/classified/blurry",
    }
    renamed_base_dir = "C:/Users/Li Mu/Desktop/copy"  # 重命名后的图像存储基础路径
    degraded_base_dir = "C:/Users/Li Mu/Desktop/degraded_images"  # 退化图像存储基础路径

    process_and_rename_images(input_dirs, renamed_base_dir, degraded_base_dir)
