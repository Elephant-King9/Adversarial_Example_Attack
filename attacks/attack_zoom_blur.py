import numpy as np
import torch
from log_config import logger
from scipy.ndimage import zoom as scizoom


# 高斯模糊攻击
class attack_zoom_blur:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()

        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.33, 0.03)][epsilon - 1]

        image = (np.array(image)).astype(np.float32)
        out = np.zeros_like(image)
        for zoom_factor in c:
            # print(zoom_factor)
            tmp = self.clipped_zoom(image, zoom_factor)
            # print("tmp:", tmp.shape)
            out = tmp + out

        image = (image + out) / (len(c) + 1)
        perturbed_image = np.clip(x, 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        logger.debug(f'perturbed_image shape{perturbed_image.shape}')
        return perturbed_image

    def clipped_zoom(self, img, zoom_factor):
        h = img.shape[0]
        w = img.shape[1]
        # print("h:",h)
        # print("w:",w)

        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))
        cw = int(np.ceil(w / zoom_factor))
        # print("ch:",ch)
        # print("cw:",cw)

        top1 = (h - ch) // 2
        top2 = (w - cw) // 2
        img = scizoom(img[top1:top1 + ch, top2:top2 + cw], (zoom_factor, zoom_factor, 1), order=1)
        # print("img:", img.shape)
        # trim off any extra pixels
        trim_top1 = (img.shape[0] - h) // 2
        trim_top2 = (img.shape[1] - w) // 2

        temp = img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]
        # print("temp:", temp.shape)

        return img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]

