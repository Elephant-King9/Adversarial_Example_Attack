import os

import torchvision.models
from torch import nn
from torchvision.models import ResNet50_Weights

# 加载预训练的 ResNet-50 模型
resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

print(resnet50)
resnet50.add_module('add_linear', nn.Linear(1000, 10))