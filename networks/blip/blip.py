"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import warnings

warnings.filterwarnings("ignore")

from .vit import VisionTransformer, interpolate_pos_embed
from .med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import os

os.environ['HTTP_PROXY'] = "http://192.168.1.10:7890"
os.environ['HTTPS_PROXY'] = "http://192.168.1.10:7890"

import torch
from torch import nn

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

# 自定义位置的bert分词器预训练地址
bert_tokenizer_path = "././assets/Pre-training_files/BERT/bert-base-uncased"


# 编码器部分
class BLIP_Base(nn.Module):
    # 初始化
    def __init__(self,
                 # 杨师姐这里的config路径应该是在使用的时候传入的，不是默认的路径
                 # 源码里是这个路径，但是杨师姐代码里是吧med_config.json文件放在了同目录下，我这里按照源码为准
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        # 创建Vit模型
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        # 初始化BERT，并且获得tokenizer(分词器)
        self.tokenizer = init_tokenizer()
        # 作用是从一个 med_config(JSON) 文件中加载 BERT 模型的配置参数，并实例化一个 BertConfig 对象
        # med_config 是传入的参数，为JSON文件的路径
        med_config = BertConfig.from_json_file(med_config)
        # vision_width为create_vit返回的参数，将对象中的encoder_width设为获取Vit模型的vision_width
        med_config.encoder_width = vision_width
        # BertModel是自定义的类(med.py)，是一个基础的 BERT 模型，只包含了 BERT 的编码器部分
        # 初始化一个BERT模型，使用med_config配置，不添加池化层

        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):
        """

        :param image: Tensor图片
        :param caption:表示文本输入的字符串。
        :param mode:指定操作模式的字符串（'image', 'text', 或 'multimodal'）
        :return:表示给定模式的输出特征的张量
        """
        # 输入的模型只能是: 图像、文字、多模态
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"

        # 将caption输入分词器进行划分，并且返回值为tensor张量
        # return_tensors="pt" 代表返回Pytorch Tensor张量
        # text代表字典生成这个句子的编码矩阵
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == 'image':
            # 将图像通过Vit编码器转化为image_embeds
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            # input_ids将分词器(tokenizer)拆分成的更小的单元(token)映射到唯一一个索引
            # attention_mask 是用于在注意力机制中指示哪些 token 需要被注意的掩码。具体来说，它用于区分实际的文本 token 和填充的 token（padding token）
            # 使用 return_dict=True 可以更方便地访问编码器的多个输出
            # mode='text' 这个参数指定编码器的模式为文本模式。
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            # 获取图像Embedding
            image_embeds = self.visual_encoder(image)
            # 初始化Attention中的𝛼
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            # 每个序列的第一个 token 设置为一个特定的编码器 token
            # 类似于生成[CLS]?
            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            # 将图像文本结合进行编码
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            # 它代表了编码器在处理完输入序列（包括文本和图像特征）后的最后隐藏状态
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        # 与BLIP_Base基本相同，修改了text_encoder->text_decoder，多了prompt，作用会在代码中的注释讲解
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        # 创建方法相同，也是返回Encoder
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        # 同Base，创建分词器
        self.tokenizer = init_tokenizer()
        # 同Base，配置文件
        med_config = BertConfig.from_json_file(med_config)
        # 同Base，设置encoder维度
        med_config.encoder_width = vision_width
        # 与Base不同，调用的方法不同
        # BertLMHeadModel 在 BertModel 的基础上添加了语言模型头，专用于语言建模任务，适合生成或补全文本
        self.text_decoder = BertLMHeadModel(config=med_config)

        # prompt的作用是引导文本生成，先给文本一个开头，让文本生成可以根据认为提供的开头来更好的生成后续文本
        self.prompt = prompt
        # 调用分词器（tokenizer）将 prompt 转换为 token IDs
        # input_ids 用于获得文字根据字典的编码序列，Eg. [101, 7592, 1010, 2129, 2024, 2017, 102]
        # 减去 1 通常是为了排除特殊 token 的影响，例如 [CLS] 或 [SEP] token
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):
        # 不用区分三种情况了
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


# 用于初始化一个 BERT 分词器（Tokenizer），并为其添加一些特殊的标记（tokens）
def init_tokenizer():
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 从预训练模型加载 BERT 分词器，填入本地的文件路径
    # 这个地方是杨师姐自己配置的路径
    # tokenizer = BertTokenizer.from_pretrained('assets/bert-base-uncased/')
    # 这是我要配置的路径
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    # 好像是bert分段链接的两个标记
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    # 获取DEC标记
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    # 返回的类型为transformers.BertTokenizer
    return tokenizer


# 用于创建 Vision Transformer (ViT) 模型
def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    """
    :param vit: 指定要创建的 ViT 模型的大小，可以是 'base' 或 'large'
    :param image_size: 输入图像的尺寸。
    :param use_grad_checkpointing: 是否使用渐进层激活检查点，以节省内存（默认值为 False）
    :param ckpt_layer: 指定从哪个层开始使用激活检查点（默认值为 0）
    :param drop_path_rate: 随机路径丢弃的比例，用于防止过拟合（默认值为 0）
    :return:
    """

    # 断言语句，用于判断vit必须在这两个参数之间选择
    assert vit in ['base', 'large'], "vit parameter must be base or large"

    # 创建base大小的vit
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    # 创建large大小的vit
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    # 返回Encoder: 具体的视觉编码器模型
    # 返回vision_width:转换为向量的维度大小
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
