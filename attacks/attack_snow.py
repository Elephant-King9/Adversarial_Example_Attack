import cv2
import numpy as np
import torch
from log_config import logger
from scipy.ndimage import zoom as scizoom
from PIL import Image as PILImage
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary



# 散焦模糊攻击
class attack_snow:
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

        if h > 224 and w > 224:
            img_h, img_w = np.random.randint(0, h - 224), np.random.randint(0, w - 224)
        else:
            img_h, img_w = 0, 0
        # x = x[img_h:img_h + 224, img_w:img_w + 224][..., [2, 1, 0]]
        image = image[img_h:img_h + min(224, h), img_w:img_w + min(224, w)][..., [2, 1, 0]]
        # print("x:", x.shape)

        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

        image = np.array(image, dtype=np.float32) / 255.
        snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        snow_layer = self.clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        if h > 224 and w > 224:
            image = c[6] * image + (1 - c[6]) * np.maximum(image,
                                                           cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(224, 224,
                                                                                                           1) * 1.5 + 0.5)
        else:
            image = c[6] * image + (1 - c[6]) * np.maximum(image,
                                                           cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(min(224, h),
                                                                                                           min(224, w),
                                                                                                           1) * 1.5 + 0.5)

        perturbed_image = np.clip(image + snow_layer + np.rot90(snow_layer, k=2), 0, 1)
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
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


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0, **kwargs):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
