import os
import numpy as np
import random
import imageio
import cv2
from tqdm import tqdm
import pandas as pd
import shutil

# 模糊退化函数
def apply_blur(img, blur_level=5):
    """对输入图像应用高斯模糊"""
    return cv2.GaussianBlur(img, (blur_level, blur_level), 0)

# 特征提取函数
def extract_features(img):
    """提取图像统计特征"""
    brightness = img.mean()
    rgb_mean = img.mean(axis=(0, 1))
    rgb_std = img.std(axis=(0, 1))
    laplace_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return brightness, rgb_mean, rgb_std, laplace_var

# 生成假深度图
def generate_fake_depth(image_shape):
    height, width = image_shape[:2]
    xv, yv = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    depth = 1 - np.sqrt(xv**2 + yv**2)  # 中心为最近，边缘为最远
    return depth

def main(rgb_path, output_path):
    # 定义退化类型参数
    N_lambda = {
        "blue_shift": [0.2, 0.4, 0.9],   # 蓝偏
        "green_shift": [0.3, 0.9, 0.5], # 绿偏
        "low_light": [0.5, 0.5, 0.5],   # 低光
    }

    # 创建退化图像保存目录
    for folder in N_lambda.keys():
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    os.makedirs(os.path.join(output_path, "blur"), exist_ok=True)

    # 定义原始文件保存目录
    original_save_folder = os.path.join(output_path, "new")
    if not os.path.exists(original_save_folder):
        print(f"Error: 原始文件夹 {original_save_folder} 不存在。")
        return

    # 定义退化标签
    water_type_label = {
        "blue_shift": "blue_shift",
        "green_shift": "green_shift",
        "low_light": "low_light",
        "blur": "blur",
    }

    # 获取 RGB 图像文件
    rgb_files = [os.path.join(rgb_path, f) for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    feature_list = []

    # 遍历 RGB 图像
    for idx, rgb_file in tqdm(enumerate(rgb_files)):
        # 检查是否已存在复制的原文件
        original_img_name = os.path.join(original_save_folder, f"{idx}.png")
        if not os.path.exists(original_img_name):
            shutil.copy(rgb_file, original_img_name)

        org_img = imageio.imread(rgb_file) / 255.0
        if len(org_img.shape) == 2:
            org_img = np.stack([org_img] * 3, axis=-1)

        # 生成假深度图
        org_depth = generate_fake_depth(org_img.shape)

        # 遍历退化类型
        for water_type in N_lambda.keys():
            rand_num = 6
            for i in range(rand_num):
                depth = np.random.uniform(0.5, 15) * org_depth

                # 计算传输率 T_x
                T_x = np.ndarray((org_img.shape[0], org_img.shape[1], 3))
                T_x[:, :, 0] = N_lambda[water_type][2] * depth
                T_x[:, :, 1] = N_lambda[water_type][1] * depth
                T_x[:, :, 2] = N_lambda[water_type][0] * depth
                T_x = (T_x - T_x.min()) / (T_x.max() - T_x.min())

                # 背景光 B_lambda
                B_lambda = np.ndarray((org_img.shape[0], org_img.shape[1], 3))
                B_lambda[:, :, 0].fill(1.5 * N_lambda[water_type][2])
                B_lambda[:, :, 1].fill(1.5 * N_lambda[water_type][1])
                B_lambda[:, :, 2].fill(1.5 * N_lambda[water_type][0])

                # 生成退化图像
                img = org_img * T_x + B_lambda * (1 - T_x)
                img = (img - img.min()) / (img.max() - img.min())

                # 保存图像
                save_folder = os.path.join(output_path, water_type_label[water_type])
                img_name = os.path.join(save_folder, f"{idx}_{i}.png")
                imageio.imwrite(img_name, (img * 255).astype(np.uint8))

                # 提取特征
                brightness, rgb_mean, rgb_std, laplace_var = extract_features(img)
                features = {
                    "filename": img_name,
                    "type": water_type_label[water_type],
                    "brightness": brightness,
                    "rgb_mean": rgb_mean.tolist(),
                    "rgb_std": rgb_std.tolist(),
                    "laplace_var": laplace_var,
                }
                feature_list.append(features)

        # 单独处理模糊退化
        rand_num = 6
        for i in range(rand_num):
            blur_img = apply_blur(org_img, blur_level=random.choice([3, 5, 7]))
            save_folder = os.path.join(output_path, "blur")
            img_name = os.path.join(save_folder, f"{idx}_{i}.png")
            imageio.imwrite(img_name, (blur_img * 255).astype(np.uint8))

            # 提取特征
            brightness, rgb_mean, rgb_std, laplace_var = extract_features(blur_img)
            features = {
                "filename": img_name,
                "type": "blur",
                "brightness": brightness,
                "rgb_mean": rgb_mean.tolist(),
                "rgb_std": rgb_std.tolist(),
                "laplace_var": laplace_var,
            }
            feature_list.append(features)

    # 保存特征到 CSV 文件
    df = pd.DataFrame(feature_list)
    df.to_csv(os.path.join(output_path, "image_features.csv"), index=False)

if __name__ == "__main__":
    rgb_path = "C:/Users/Li Mu/Desktop/yuanshi"  # 替换为输入图像路径
    output_path = "D:/python_project/APMCM/newgenerate"  # 替换为输出目录路径
    main(rgb_path, output_path)
