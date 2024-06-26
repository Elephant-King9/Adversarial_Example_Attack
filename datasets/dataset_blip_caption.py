import json
import os
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from loguru import logger

# 先直接指定路径测试，后期再改
data_path = '././assets/datasets/COCO/annotations/captions_val2014.json'


# 根据图片ID获取图片的路径在get_filename中配置
# 目前仅配置了coco数据集的路径

# 被model_caption引用
class DatasetCaption(Dataset):

    def __init__(self, config, data_dir: str):
        # data_dir暂时没用
        self.config = config
        self.dataset = config.dataset
        self.device = config.device

        # 用于预处理图片，分为标准化和非标准化两种方式
        # 类似于定义transforms.Compose
        # 也就是定义对图片预处理的一套方法
        self.preprocess = self.get_preprocess(self.config.blip_image_size)
        self.preprocess_unnorm = self.get_preprocess(self.config.blip_image_size, norm=False)
        # 获取标准化参数
        self.norm = self.get_norm()
        # 找到对应数据集的.json文件
        # 这里我没有修改路径，先在类顶端自己定义后面再说
        # data_path = data_dir + '/' + self.config.data_name + '.json'

        # 打开JSON文件
        # JSON 结构中有一个主键 annotations，其中包含了所有的图像注解
        with open(data_path, 'r') as f:
            data = json.load(f)['annotations']
        # 获取图片的id和图片描述
        image_ids = [d['image_id'] for d in data]
        captions = [d['caption'] for d in data]

        self.image_ids = []
        self.image_id_to_captions = dict()
        # zip(image_ids, captions)同时遍历图像 ID 列表和描述列表
        for image_id, caption in zip(image_ids, captions):
            # 当前图片还没有被添加
            if image_id not in self.image_id_to_captions.keys():
                self.image_ids.append(image_id)
                self.image_id_to_captions[image_id] = []
            self.image_id_to_captions[image_id].append(caption)
        # 输出获取的数据集中的数据大小
        # print("Data size is %0d" % len(self.image_ids))
        logger.info(f"Data size is {len(self.image_ids)}")

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # 根据图片在数据集中的下标获取ID
        image_id = self.image_ids[item]
        # 根据图片ID找到对应的图片
        filename = self.get_filename(image_id)
        # 将图片转化为PIL格式
        image_pil = Image.open(filename)
        # 对图像进行预处理
        image = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        # 进行非标准化的处理，用于attack，这么做就不用标准化了
        image_unnorm = self.preprocess_unnorm(image_pil)  # for attack
        # 获取图片的标注信息
        captions = self.image_id_to_captions[image_id]
        # print(image_id, image.shape, image_unnorm.shape, captions)
        """
        返回信息为
            图片ID
            经过预处理的Tensor类型图片(标准化)
            经过预处理的Tensor类型图片(非标准化)
            图像描述
        """
        return image_id, image, image_unnorm, captions

    def __len__(self) -> int:
        return len(self.image_ids)

    # 根据图片id获取图片
    def get_filename(self, image_id):
        # 仅配置了coco数据集，其他数据集后续在修改
        if self.dataset == 'coco':
            # 师姐的路径
            # filename = f"./data/coco/train2014/COCO_train2014_{int(image_id):012d}.jpg"
            # 我的路径
            filename = f"././assets/datasets/COCO/train2014/COCO_train2014_{int(image_id):012d}.jpg"
            if not os.path.isfile(filename):
                # 师姐的路径
                # filename = f"./data/coco/val2014/COCO_val2014_{int(image_id):012d}.jpg"
                # 我的路径
                filename = f"././assets/datasets/COCO/val2014/COCO_val2014_{int(image_id):012d}.jpg"
        elif self.dataset == 'flickr8k':
            filename = "./data/flickr8k/Flickr8k_Dataset/%s" % image_id
        elif self.dataset == 'flickr30k':
            filename = "./data/flickr30k/flickr30k-images/%s" % image_id
        else:
            # print('error data: %s' % self.dataset)
            # 数据集数据不正确
            logger.warning('error data: %s' % self.dataset)
            exit()
        return filename

    def get_preprocess(self, image_size=480, norm=True):
        # 用于图像预处理

        # 转化图片为RGB格式
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        # 转化为numpy数组
        def _convert_image_to_arr(image):
            return np.array(image)

        # 判断是否标准化
        if norm:
            preprocess = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=TF.InterpolationMode.BICUBIC),
                _convert_image_to_rgb,
                ToTensor(),
                # 测试图像标准化，先注释掉
                # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=TF.InterpolationMode.BICUBIC),
                _convert_image_to_rgb,
                _convert_image_to_arr,
            ])
        return preprocess

    def get_norm(self):
        # 获取标准化参数
        return Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
