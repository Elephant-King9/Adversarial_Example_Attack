import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils.mkdir import mkdir
from log_config import logger


def save_image_comparison(config, adv_examples, eps):
    num_examples = len(adv_examples)

    # 创建保存图像的目录
    adv_dir = os.path.join(config.compare_path, config.attack)
    mkdir(adv_dir)
    adv_dir = os.path.join(adv_dir, config.dataset)
    mkdir(adv_dir)
    adv_dir = os.path.join(adv_dir, str(eps))
    mkdir(adv_dir)

    for i, (init_pred, final_pred, orig_img, adv_img) in enumerate(adv_examples):
        # 归一化图像到0-255范围
        orig_img = (orig_img * 255).astype(np.uint8)
        adv_img = (adv_img * 255).astype(np.uint8)

        # 确保图像数据在 [0, 255] 范围内
        orig_img = np.clip(orig_img, 0, 255)
        adv_img = np.clip(adv_img, 0, 255)

        # 如果是三通道图像，则转置通道
        if orig_img.ndim == 3 and orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
        if adv_img.ndim == 3 and adv_img.shape[0] == 3:
            adv_img = np.transpose(adv_img, (1, 2, 0))

        # 创建对比图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title(f'Original (Pred: {init_pred})')
        axes[0].axis('off')

        axes[1].imshow(adv_img, cmap='gray')
        axes[1].set_title(f'Adversarial (Pred: {final_pred})')
        axes[1].axis('off')

        # 调整子图之间的间距
        plt.tight_layout(pad=2.0)

        # 保存对比图像
        compare_img_path = os.path.join(adv_dir, f"{i}_comparison.png")
        plt.savefig(compare_img_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        logger.info(f"Comparison image {i} saved to {compare_img_path}")
