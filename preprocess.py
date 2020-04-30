import torch
import torchvision.transforms as transforms

from image_processing import *

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = normalize or __imagenet_stats

    if name == 'imagenet':
        scale_size = scale_size or 256
        input_size = input_size or 224
        if augment:
            return inception_preproccess(input_size, normalize=normalize)
        else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    elif 'cifar' in name:
        input_size = input_size or 32
        if augment:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

    elif name == 'svhn':
        input_size = input_size or 32
        if augment:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

    elif name == 'mnist':

        return transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])
                   
        # normalize = {'mean': [0.5], 'std': [0.5]}
        # input_size = input_size or 28
        # if augment:
        #     scale_size = scale_size or 32
        #     return pad_random_crop(input_size, scale_size=scale_size,
        #                            normalize=normalize)
        # else:
        #     scale_size = scale_size or 32
        #     return scale_crop(input_size=input_size,
        #                       scale_size=scale_size, normalize=normalize)
