import torch
from .ALA_lib import RGB2Lab_t, Lab2RGB_t, light_filter, update_paras
from torch.utils.tensorboard import SummaryWriter
from log_config import logger


class attack_ALA_caption:
    def __init__(self, model, config):
        self.config = config
        self.device = config.device
        self.model = model
        self.batch_size = config.batch_size
        self.segment = config.segment
        self.init_range = config.init_range
        self.transform = config.transform
        self.tau = config.tau
        self.eta = config.eta
        self.lr = config.lr
        # self.maxIters = config.maxIters
        # self.count = 0
        # self.writer = SummaryWriter(self.config.dirname)  # tensorboard

    def attack(self, image, epsilon, annotations, **kwargs):
        image_id = kwargs.get('image_id')
        # adv_images = []
        # self.count += 1
        # self.write_log = True if self.count <= 10 else False

        adv_image = self.ALA_method(image_id, image, annotations, self.segment, self.eta, epsilon)
        # adv_images.append(adv_image)
        return adv_image

    def ALA_method(self, image_id, image, annotations, segment, eta, epsilon):
        # adv_image = image
        # image = self.model.dataset.to_image(image)
        perturbed_image = image.clone().detach().to(self.device)
        perturbed_image = perturbed_image.squeeze(0)
        perturbed_image = perturbed_image.permute(1, 2, 0)
        X_ori = (RGB2Lab_t(perturbed_image / 1.0) + 128) / 255.0
        X_ori = X_ori.unsqueeze(0).type(torch.FloatTensor).to(self.device)

        light, color = torch.split(X_ori, [1, 2], dim=1)
        light_max = torch.max(light, dim=2)[0].max(dim=2)[0]
        light_min = torch.min(light, dim=2)[0].min(dim=2)[0]

        color = color.to(self.device)
        light = light.to(self.device)

        Paras_light = torch.ones(self.batch_size, 1, segment).to(self.device)
        Paras_light.requires_grad = True

        max_loss = 0
        adv_image = perturbed_image
        for itr in range(epsilon):
            X_adv_light = light_filter(light, Paras_light, segment, light_max.to(self.device),
                                       light_min.to(self.device))
            X_adv = torch.cat((X_adv_light, color), dim=1) * 255.0
            X_adv = Lab2RGB_t(X_adv.squeeze(0) - 128).div(255).clamp_(0, 1).unsqueeze(0)

            # 关键求损失的方法
            adv_loss, paras_loss, loss = self.calc_loss(image_id, X_adv, annotations,
                                                        Paras_light, segment, eta)

            logger.debug(f'adv_loss: {adv_loss}')
            # if self.write_log:
            #     self.writer.add_scalar(f'adv_loss_{self.plot_pic_idx}', adv_loss, itr)
            #     self.writer.add_scalar(f'para_loss_{self.plot_pic_idx}', paras_loss, itr)
            #     self.writer.add_scalar(f'loss_{self.plot_pic_idx}', loss, itr)

            if adv_loss > max_loss:
                max_loss = adv_loss
                adv_image = X_adv.detach()

        # return adv_image
        if epsilon != 0:
            return adv_image
        else:
            perturbed_image = adv_image.permute(2, 0, 1)
            perturbed_image = perturbed_image.unsqueeze(0)
            return perturbed_image
    def calc_loss(self, image_id, image, annotations, Paras_light, segment, eta):
        # if self.config. == "image_caption":
        #     return self.forward_and_update_paras_image_caption(image_id, image, annotations, Paras_light, segment, eta)
        # elif self.config.task_name == "vqa":
        #     return self.forward_and_update_paras_vqa(image_id, image, annotations, Paras_light, segment, eta)
        # else:
        #     print(f"Error task_name: {self.config.task_name}")
        #     exit()
        return self.forward_and_update_paras_image_caption(image_id, image, annotations, Paras_light, segment, eta)

    def forward_and_update_paras_image_caption(self, image_id, image, annotations, Paras_light, segment, eta):
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
        return mean_adv_loss, mean_para_loss, mean_loss

    # def forward_and_update_paras_vqa(self, image_id, image, annotations, Paras_light, segment, eta):
    #     question_ids, questions, answers = annotations
    #     if image.ndim == 5:
    #         image = image.squeeze(0)
    #
    #     num, mean_adv_loss, mean_para_loss, mean_loss = 0, 0, 0, 0
    #     for question, answer in zip(questions, answers):
    #         for a in answer:
    #             num += 1
    #             self.model.model.zero_grad()
    #             adv_loss = self.model.calc_one_sample_loss(image_id, image, question[0], a[0])
    #             paras_loss = 1 - torch.abs(Paras_light).sum() / segment
    #             loss = -adv_loss + eta * paras_loss
    #             loss.backward(retain_graph=True)
    #             update_paras(Paras_light, self.lr, self.batch_size)
    #             mean_adv_loss += adv_loss.detach()
    #             mean_para_loss += paras_loss.detach()
    #             mean_loss += loss.detach()
    #     mean_adv_loss /= num
    #     mean_para_loss /= num
    #     mean_loss /= num
    #     return mean_adv_loss, mean_para_loss, mean_loss

    # def close_writer(self):
    #     self.writer.close()
