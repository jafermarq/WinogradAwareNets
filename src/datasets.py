import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

def cifar10Transformation(is_training, doAugment=True):
    if is_training:
        if doAugment:
            return  transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            return  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    else:
        return  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


def getCIFAR10(batch_size, use_cuda, path_to_data='./data', val_size= 0.1):

    # if pin memory --> very high CPU usage: see Pytorch issue #25010
    # With False, it makes training faster on CIFAR10 (at least for reasonabled batch_size size). For Imagenet, we train faster by seting pin_memory=True
    kwargs = {'num_workers': 4, 'pin_memory': False,  'drop_last': True} if use_cuda else {'drop_last': True}

    train_dataset = datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=cifar10Transformation(True))
    test_dataset = datasets.CIFAR10(root=path_to_data, train=False, download=True, transform=cifar10Transformation(False))

    num_train = len(train_dataset)
    split = int(np.floor(val_size * num_train))

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[num_train - split,split])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader