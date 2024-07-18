import torch
from torchvision import transforms
from attacks.ALA_lib import RGB2Lab_t, Lab2RGB_t, light_filter, update_paras
from log_config import logger


class attack_ALA_caption:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.segment = config.segment
        self.init_range = config.init_range
        self.transform = config.transform
        self.tau = config.tau
        self.eta = config.eta
        self.lr = config.lr

    def attack(self, image_id, image, annotations, Paras_light, segment, eta, epsilon):
        captions = [caption[0] for caption in annotations]
        if image.ndim == 5:
            image = image.squeeze(0)

        num, mean_adv_loss, mean_para_loss, mean_loss = 0, 0, 0, 0
        for caption in captions:
            num += 1
            self.model.model.zero_grad()
            adv_loss = self.model.calc_one_sample_loss(image_id, image, caption)
            paras_loss = 1 - torch.abs(Paras_light).sum() / segment
            loss = -adv_loss + eta * paras_loss
            loss.backward(retain_graph=True)
            update_paras(Paras_light, self.lr, self.batch_size)
            mean_adv_loss += adv_loss.detach()
            mean_para_loss += paras_loss.detach()
            mean_loss += loss.detach()
        mean_adv_loss /= num
        mean_para_loss /= num
        mean_loss /= num

        # ALA caption 特有的处理
        # 根据需要进一步处理结果或返回适当的数据结构
        return mean_adv_loss, mean_para_loss, mean_loss
