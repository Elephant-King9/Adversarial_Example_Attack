import torchvision

path = "../assets/datasets"
torchvision.datasets.CIFAR10(root=path, train=True, download=True)