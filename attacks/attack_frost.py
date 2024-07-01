import cv2
import numpy as np
import torch
from log_config import logger


# 霜攻击
class attack_frost:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):

        image = image * 255
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        logger.debug(f'image shape before:{image.shape}')  # image shape:(1, 3, 480, 480)
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))
        logger.debug(f'image shape after:{image.shape}')  # image shape after:(480, 480, 3)

        h = image.shape[0]
        w = image.shape[1]

        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][epsilon - 1]
        idx = np.random.randint(5)
        # print("idx:",idx)
        # filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
        filename = ['./assets/frost_img/frost1.png', './assets/frost_img/frost2.png', './assets/frost_img/frost3.png',
                    './assets/frost_img/frost4.jpg', './assets/frost_img/frost5.jpg', './assets/frost_img/frost6.jpg'][
            idx]
        # print("filename:",filename)
        frost = cv2.imread(filename)
        # print("frost:", frost.shape)

        h_f = frost.shape[0]
        w_f = frost.shape[1]

        if h > h_f and w > w_f:
            img_h, img_w = np.random.randint(0, h - h_f), np.random.randint(0, w - w_f)
            image = image[img_h:img_h + h_f, img_w:img_w + w_f][..., [2, 1, 0]]
            x_start, y_start = 0, 0
            frost = frost[x_start:x_start + h_f, y_start:y_start + w_f][..., [2, 1, 0]]

        elif h > h_f and w <= w_f:
            img_h, img_w = np.random.randint(0, h - h_f), 0
            image = image[img_h:img_h + h_f, img_w:img_w + w][..., [2, 1, 0]]
            x_start, y_start = 0, 0
            frost = frost[x_start:x_start + h_f, y_start:y_start + w][..., [2, 1, 0]]

        elif h <= h_f and w > w_f:
            img_h, img_w = 0, np.random.randint(0, w - w_f)
            image = image[img_h:img_h + h, img_w:img_w + w_f][..., [2, 1, 0]]
            x_start, y_start = 0, 0
            frost = frost[x_start:x_start + h, y_start:y_start + w_f][..., [2, 1, 0]]

        else:
            img_h, img_w = 0, 0
            image = image[img_h:img_h + h, img_w:img_w + w][..., [2, 1, 0]]
            x_start, y_start = 0, 0
            frost = frost[x_start:x_start + h, y_start:y_start + w][..., [2, 1, 0]]

        perturbed_image = np.clip(c[0] * np.array(image) + c[1] * frost, 0, 255) / 255
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image
