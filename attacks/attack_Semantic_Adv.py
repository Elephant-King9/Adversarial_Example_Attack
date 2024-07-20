import matplotlib
import numpy as np
import torch

from log_config import logger


class attack_Semantic_Adv:
    def __init__(self, model, config):
        self.model = model
        self.device = config.device

    def attack(self, image, epsilon, label, **kwargs):
        perturbed_image = image.clone().detach().to(self.device)
        perturbed_image = perturbed_image.squeeze(0)
        perturbed_image = perturbed_image.permute(1, 2, 0).cpu().numpy()
        # 转化为HSV空间
        X_hsv = matplotlib.colors.rgb_to_hsv(perturbed_image)
        X = perturbed_image
        for i in range(epsilon):
            X_adv_hsv = np.copy(X_hsv)
            d_h = np.random.uniform(0, 1, size=(X_adv_hsv.shape[0], X_adv_hsv.shape[1]))
            d_s = np.random.uniform(-1, 1, size=(X_adv_hsv.shape[0], X_adv_hsv.shape[1])) * float(i) / epsilon

            X_adv_hsv[:, :, 0] = (X_hsv[:, :, 0] + d_h) % 1.0
            X_adv_hsv[:, :, 1] = np.clip(X_hsv[:, :, 1] + d_s, 0., 1.)

            X = matplotlib.colors.hsv_to_rgb(X_adv_hsv)
            X = np.clip(X, 0., 1.)
        logger.debug(f'Adversarial examples shape:{X.shape}')
        X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return X



