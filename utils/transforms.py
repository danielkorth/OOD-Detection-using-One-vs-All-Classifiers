# Common Augmentations for the datasets

import torch
import torchvision.transforms as T

CIFAR10_MEAN = torch.Tensor([0.491, 0.482, 0.446])
CIFAR10_STD = torch.Tensor([0.247, 0.243, 0.261])

CIFAR100_MEAN = torch.Tensor([0.507, 0.487, 0.441])
CIFAR100_STD = torch.Tensor([0.267, 0.256, 0.276])

IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225])

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

grayscale_to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1))

# MNIST


def get_mnist_transform(training=True):
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    return {'train_transform': train_transform if training else test_transform, 'test_transform': test_transform}


# CIFAR10
def get_cifar10_transform(training=True):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    return {'train_transform': train_transform if training else test_transform, 'test_transform': test_transform}


# CIFAR100
def get_cifar100_transform(training=True):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    return {'train_transform': train_transform if training else test_transform, 'test_transform': test_transform}
