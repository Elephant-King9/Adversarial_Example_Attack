from abc import ABC, abstractmethod


# 抽象类，用于规范model类的实现
class model_base(ABC):
    @abstractmethod
    def calc_loss(self, image, label):
        pass

    @abstractmethod
    def calc_image_grad(self, image, label):
        pass

