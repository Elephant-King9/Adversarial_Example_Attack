import torch
import torchvision.transforms as transforms


class test_Config:
    def __init__(self):
        self.train_gpu = '1'
        self.device = torch.device('cuda:' + self.train_gpu if torch.cuda.is_available() else 'cpu')
        self.model = 'MNIST'
        self.download = False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.batch_size = 1
        self.shuffle = True
        self.epsilons = [0, .05, .1, .15, .2, .25, .3]
        self.accuracies = []
        self.examples = []
        self.plt_path = 'results/plt_pics'
        self.adv_path = 'results/adv_pics'
        self.alpha = 1 / 75
        self.momentum = 0.9
