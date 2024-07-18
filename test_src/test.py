"""
Author: senereone
Date: 2024-06-10 15:09:05
Desc: https://github.com/Huang-yihao/ALA.git
"""

import torch
from .ALA_lib import RGB2Lab_t, Lab2RGB_t, light_filter, update_paras
from torch.utils.tensorboard import SummaryWriter


class AttackMethod(object):
    def __init__(self, config, model):
        self.config = config
        self.device = config.device
        self.model = model
        self.batch_size = 1
        self.attack_keys = [
            "ALA_para_%d_%.2f" % (segment, eta)
            for (segment, eta) in self.config.ALA_para_list
        ]
        self.maxIters = 30
        self.lr = 0.5

        self.count = 0
        self.writer = SummaryWriter(self.config.dirname)  # tensorboard

    def attack(self, image_id, image, annotations):
        adv_images = []
        # 记录攻击次数
        self.count = self.count + 1
        # 攻击次数小于10次的log
        self.write_log = True if self.count <= 10 else False

        for idx, (segment, eta) in enumerate(self.config.ALA_para_list):
            self.plot_pic_idx = idx  # log画图的index
            # ALA_method用于完成攻击方法
            adv_image = self.ALA_method(image_id, image, annotations, segment, eta)
            adv_images.append(adv_image)
        return adv_images

    def ALA_method(self, image_id, image, annotations, segment, eta):
        adv_image = image
        # 转化图片格式
        image = self.model.dataset.to_image(image)

        X_ori = (RGB2Lab_t(image / 1.0) + 128) / 255.0
        X_ori = X_ori.unsqueeze(0).type(torch.FloatTensor)

        # L channel and a,b channel
        light, color = torch.split(X_ori, [1, 2], dim=1)
        # 两次最大值计算，比一次多保留了一个维度
        light_max = torch.max(light, dim=2)[0]
        light_max = torch.max(light_max, dim=2)[0]
        light_min = torch.min(light, dim=2)[0]
        light_min = torch.min(light_min, dim=2)[0]
        color = color.cuda()
        light = light.cuda()

        Paras_light = torch.ones(self.batch_size, 1, segment).to(self.device)
        Paras_light.requires_grad = True

        max_loss = 0
        for itr in range(self.maxIters):
            # modifys the lightness
            X_adv_light = light_filter(
                light, Paras_light, segment, light_max.cuda(), light_min.cuda()
            )

            X_adv = torch.cat((X_adv_light, color), dim=1) * 255.0
            X_adv = Lab2RGB_t(X_adv.squeeze(0) - 128)
            cur_adv_image = X_adv.div(255).clamp_(0, 1).unsqueeze(0)
            adv_loss, paras_loss, loss = self.calc_loss(
                image_id,
                self.model.dataset.norm(cur_adv_image),
                annotations,
                Paras_light,
                segment,
                eta,
            )

            if self.write_log:
                self.writer.add_scalar(f'adv_loss_{self.plot_pic_idx}', adv_loss, itr)
                self.writer.add_scalar(f'para_loss_{self.plot_pic_idx}', paras_loss, itr)
                self.writer.add_scalar(f'loss_{self.plot_pic_idx}', loss, itr)

            if adv_loss > max_loss:
                max_loss = adv_loss
                adv_image = cur_adv_image.detach()
            # print(i, adv_loss, paras_loss)
        return adv_image

    def calc_loss(self, image_id, image, annotations, Paras_light, segment, eta):
        if self.config.task_name == "image_caption":
            adv_loss, paras_loss, loss = self.forward_and_update_paras_image_caption(
                image_id, image, annotations, Paras_light, segment, eta
            )
        elif self.config.task_name == "vqa":
            adv_loss, paras_loss, loss = self.forward_and_update_paras_vqa(
                image_id, image, annotations, Paras_light, segment, eta
            )
        else:
            print("error task_name: %s" % self.config.task_name)
            exit()
        return adv_loss, paras_loss, loss

    def forward_and_update_paras_image_caption(
            self, image_id, image, annotations, Paras_light, segment, eta
    ):
        # annotations为数据集中的描述
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

    def forward_and_update_paras_vqa(
            self, image_id, image, annotations, Paras_light, segment, eta
    ):
        question_ids, questions, answers = annotations
        if image.ndim == 5:
            image = image.squeeze(0)

        num, mean_adv_loss, mean_para_loss, mean_loss = 0, 0, 0, 0
        for question, answer in zip(questions, answers):
            for a in answer:
                num += 1
                self.model.model.zero_grad()
                adv_loss = self.model.calc_one_sample_loss(image_id, image, question[0], a[0])
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

    def close_writer(self):
        self.writer.close()