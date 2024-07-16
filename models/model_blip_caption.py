import torch
from torch.utils.data import DataLoader

# from ..target_model_base import TargetModelBase
from networks.blip.blip import blip_decoder
from datasets.dataset_blip_caption import DatasetCaption

from log_config import logger
from models.model_base import model_base

# 预训练模型的文件位置，从文件中提取出来便于修改
# 存放在了assets中的model_base_caption_capfilt_large.pth
pretrained_model_path = './assets/Pre-training_files/BLIP/model_base_caption_capfilt_large.pth'
# med_config.json的文件目录
med_model_path = './configs/med_config.json'
# 数据集的路径
# 先凑活这用
data_dir = './assets/datasets'


class model_blip_caption(model_base):
    def __init__(self, config):
        self.config = config
        # 这里对师姐的代码做出了修改，因为我的文件中config已经包含了batch_size了，所以修改为了在config中获取
        self.batch_size = config.batch_size
        # self.batch_size = 1
        self.device = config.device
        # self.pretrained_model_path = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
        # 预训练模型的路径
        self.pretrained_model_path = pretrained_model_path

        """
        pretrained: 预训练模型路径
        image_size: 指定输入图像的大小，在config中指出
        vit: 指定Vision Transformer为base模型
            base: 基础模型
            large:大模型
        vit_grad_ckpt: 用于控制是否在训练期间使用梯度检查点来节省内存
            False: 为不用检查
        vit_ckpt_layer: 这个参数指定了从哪一层开始使用梯度检查点
            因为vit_grad_ckpt=False，所以实际没有生效
        prompt: 这是预设的文本提示，用于图像生成任务中引导模型生成与提示相关的图像描述
        med_config: med模型的配置文件目录
            这里我设置为了我自己保存的路径
        """
        model = blip_decoder(pretrained=self.pretrained_model_path, 
                             image_size=self.config.blip_image_size, 
                             vit='base',
                             vit_grad_ckpt=False, 
                             vit_ckpt_layer=0, 
                             prompt='a picture of ',
                             med_config=med_model_path)
        model = model.to(self.device)
        self.model = model.eval()

    def get_data_loader(self):
        # 使用了dataset_caption.py
        if self.config.data_name in ['coco', 'flickr8k', 'flickr30k']:
            # 自定义路径
            # data_dir = data_dir
            # 自定义的数据集使用方式
            self.dataset = DatasetCaption(self.config, data_dir)
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
            return self.data_loader
        else:
            logger.warning('target_model.py get_data_loader task_name:%s data_name: %s not support' % (
                self.config.task_name, self.config.data_name))
            # print('target_model.py get_data_loader task_name:%s data_name: %s not support' % (
            #     self.config.task_name, self.config.data_name))
            exit()

    # 计算损失
    def calc_loss(self, clean_image, annotations):
        # 这个地方在对应的attack里面已经进行了梯度清零
        # self.model.zero_grad()
        captions = annotations
        captions = [caption[0] for caption in captions]

        if clean_image.ndim == 5:
            clean_image = clean_image.squeeze(0)
        clean_image = clean_image.expand(len(captions), -1, -1, -1)

        loss = self.model(clean_image, captions)
        return loss

    # 计算图片梯度
    def calc_image_grad(self, clean_image, annotations):
        self.model.zero_grad()
        captions = annotations
        captions = [caption[0] for caption in captions]

        if clean_image.ndim == 5:
            clean_image = clean_image.squeeze(0)
        clean_image = clean_image.clone().detach().to(self.config.device)
        clean_image = clean_image.expand(len(captions), -1, -1, -1)

        clean_image.requires_grad = True
        loss = self.calc_loss(clean_image, captions)
        loss.backward()
        # 对张量的维度取平均，并修改维度
        grad = torch.mean(clean_image.grad, 0).unsqueeze(0)
        return grad

    def predict(self, image_id, image, annotations, display=False):
        # 给图像预测caption
        # 返回图像的维度信息
        if image.ndim == 5:
            image = image.squeeze(0)
        self.model.zero_grad()

        # 打印图像张量的形状
        # print(f"Image shape before generation: {image.shape}")

        # 调用blip模型中的generate用于生成当前图像的caption
        """
        image:传入的图片
        sample:这个参数控制生成文本的方式
            False代表使用贪婪算法或 Beam Search 来生成文本，而不是随机采样
            
            注意这里需要改为True，否则无法运行
            
        num_beams： 这是 Beam Search 的参数，它设置了搜索宽度为 3
        max_length：代表生成的描述最多为20个词
        min_length：代表最少生成5个词
        [0]:在调用 generate 方法后，通常会返回一个包含多个结果的列表，每个结果对应一个生成的序列。
        通过 [0] 索引，这行代码获取这些生成序列中的第一个，通常也是评分最高（即模型认为最可能的）描述。
        """
        pred_caption = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)[0]
        # display用于判断是否在控制台输出
        if display:
            logger.info('image_id: %s, pred_caption: %s' % (image_id, pred_caption))
        return pred_caption
