from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def save_model(save_path,
               model,
               optimizer=None,
               replay_buffer=None,
               discriminator=None,
               optimizer_d=None):
    save_dict = {'model': model.state_dict()}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if replay_buffer is not None:
        save_dict['replay_buffer'] = replay_buffer
    if discriminator is not None:
        save_dict['discriminator'] = discriminator
    if optimizer_d is not None:
        save_dict['optimizer_d'] = optimizer_d
    torch.save(save_dict, save_path)


def load_model(load_path):
    checkpoint = torch.load(load_path)
    return (
        checkpoint['model'],
        checkpoint.get('optimizer', None),
        checkpoint.get('replay_buffer', None),
        checkpoint.get('discriminator', None),
        checkpoint.get('optimizer_d', None),
    )


def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert (lo < hi), f"[rescale] lo={lo} must be smaller than hi={hi}"
    old_width = torch.max(x) - torch.min(x)
    old_center = torch.min(x) + (old_width / 2.)
    new_width = float(hi - lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x


def corruption(x, type_='ebm', noise_scale=0.3):
    assert type_ in ['ebm', 'flow']
    # mask=1 if the pixel is visible
    mask = torch.zeros_like(x)
    if type_ == 'ebm':
        # Corrupt the rows 0, 2, 4, ....
        mask[..., torch.arange(0, mask.shape[-2], step=2), :] = 1
    elif type_ == 'flow':
        # Corrupt the lower part
        mask[..., :mask.shape[-2] // 2, :] = 1
    broken_data = x * mask + (1 - mask) * noise_scale * torch.randn_like(x)
    broken_data = torch.clamp(broken_data, 1e-4, 1 - 1e-4)
    return broken_data, mask


transform = transforms.Compose([
    # convert PIL image to tensor:
    transforms.ToTensor(),
    # add uniform noise:
    transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
    # rescale to [0.0001, 0.9999]:
    transforms.Lambda(lambda x: rescale(x, 0.0001, 0.9999)),
])

# image tensor range [0, 1]
train_set = MNIST(
    root="./data",
    download=True,
    transform=transform,
    train=True,
)

val_set = MNIST(
    root="./data",
    download=True,
    transform=transform,
    train=False,
)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# CIFAR dataset
cifar_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomRotation(15), 
    transforms.ToTensor(),
    AddGaussianNoise(0.0, 0.1), 
    transforms.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])
])

cifar_train_set = CIFAR10(
    root="./data",
    download=True,
    transform=cifar_transform,
    train=True,
)

cifar_val_set = CIFAR10(
    root="./data",
    download=True,
    transform=cifar_transform,
    train=False,
)


class MockResidualBlock(nn.Module):
    """ A simplified residual block used in RealNVP.

    Image feature shape is assumed to be unchanged.
    """

    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 output_activation=True):
        super(MockResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out,
                      c_out,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(c_out))
        self.skip_connection = nn.Identity()
        if c_in != c_out:
            self.skip_connection = nn.Sequential(nn.Conv2d(c_in, c_out, 1, 1),
                                                 nn.BatchNorm2d(c_out))

        self.output_activation = output_activation

    def forward(self, x):
        if self.output_activation:
            return F.relu(self.conv(x) + self.skip_connection(x), inplace=True)
        else:
            return self.conv(x) + self.skip_connection(x)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_square(n):
    if n == 0: return True
    return n == int(n**0.5)**2

def hello():
    print('Good luck!')

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().cuda()
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().cuda()

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas