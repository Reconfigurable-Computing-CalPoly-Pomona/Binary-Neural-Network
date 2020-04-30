import torch.nn as nn

#__all__ = ['vggnet']

class VGGNet_CIFAR10(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGNet_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


class VGGNet_SVHN(VGGNet_CIFAR10):
    def __init__(self, num_classes=1000):
        super(VGGNet_SVHN, self).__init__(num_classes=10)


def vggnet(**kwargs):
    num_classes, depth, dataset = map(kwargs.get, 
                                      ['num_classes', 'depth', 'dataset'])

    if dataset == 'cifar10':
        num_classes = num_classes or 10
        return VGGNet_CIFAR10(num_classes)

    if dataset == 'svhn':
        num_classes = num_classes or 10
        return VGGNet_SVHN(num_classes)
