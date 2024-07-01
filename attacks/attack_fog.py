import cv2
import numpy as np
import torch
from log_config import logger


# 散焦模糊攻击
class attack_fog:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        logger.debug(f'image shape before:{image.shape}')  # image shape:(1, 3, 480, 480)
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))
        logger.debug(f'image shape after:{image.shape}')  # image shape after:(480, 480, 3)

        h = image.shape[0]
        w = image.shape[1]
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][epsilon - 1]

        image = np.array(image)
        max_val = image.max()
        tmp = c[0] * self.plasma_fractal(wibbledecay=c[1])[:h, :w][..., np.newaxis]
        image = image + tmp
        perturbed_image = np.clip(image * max_val / (max_val + c[0]), 0, 1)
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image

    def plasma_fractal(self, mapsize=1024, wibbledecay=3):
        """
        Generate a heightmap using diamond-square algorithm.
        Return square 2d array, side length 'mapsize', of floats in range 0-255.
        'mapsize' must be a power of two.
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float_)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()
