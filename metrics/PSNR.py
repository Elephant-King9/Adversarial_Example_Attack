import numpy as np
import cv2
import torch


class PSNR(object):
    def __init__(self, original, compressed):
        self.original = original
        self.compressed = compressed

    def calculate_psnr(self, **kwargs):
        # 如果输入是 PyTorch 张量，则将其转换为 NumPy 数组
        if isinstance(self.original, torch.Tensor):
            self.original = self.original.cpu().numpy()
        if isinstance(self.compressed, torch.Tensor):
            self.compressed = self.compressed.cpu().numpy()

        # 确保图像的范围是 [0, 1]
        self.original = np.clip(self.original, 0, 1)
        self.compressed = np.clip(self.compressed, 0, 1)

        mse = np.mean((self.original - self.compressed) ** 2)
        if mse == 0:  # MSE 为 0 时，表示两张图像完全相同
            return float('inf')
        max_pixel = 1.0  # 因为图像已经归一化到 [0, 1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr


# 示例使用
if __name__ == "__main__":
    original = cv2.imread("original_image.png")
    compressed = cv2.imread("compressed_image.png")

    psnr_value = calculate_psnr(original, compressed)
    print(f"PSNR value is {psnr_value} dB")
