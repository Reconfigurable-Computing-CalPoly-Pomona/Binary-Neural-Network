import torch.nn as nn

from .binarized_modules import  BinarizeLinear, BinarizeConv2d

#__all__ = ['vgg_cifar10_binary']

class VGGNet_CIFAR10_Binary(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGNet_CIFAR10_Binary, self).__init__()
        self.infl_ratio=3
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),

            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),

            BinarizeLinear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(dim=1)
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


class VGGNet_SVHN_Binary(VGGNet_CIFAR10_Binary):
    def __init__(self, num_classes=1000):
        super(VGGNet_SVHN_Binary, self).__init__(num_classes=10)


def vggnet_binary(**kwargs):
    num_classes, depth, dataset = map(kwargs.get, 
                                      ['num_classes', 'depth', 'dataset'])

    if dataset == 'cifar10':
        num_classes = num_classes or 10
        return VGGNet_CIFAR10_Binary(num_classes)

    if dataset == 'svhn':
        num_classes = num_classes or 10
        return VGGNet_SVHN_Binary(num_classes)
