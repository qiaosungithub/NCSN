import torch
import torch.nn as nn
import torch.nn.functional as F

# construct a UNET network for the NCSN score model
# the u-net will get extra input of sigma

def conv3x3(in_c, out_c, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ConditionalInstanceNorm2dPlus(nn.Module):
    # copied from the ncsn repo
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out

class CondUNet(nn.Module):
    def __init__(self, L, activation=F.relu):
        super(CondUNet, self).__init__()

        # the input shape is 1x28x28

        self.conv1 = conv3x3(1, 16)
        self.conv2 = conv3x3(16, 64)
        self.conv3 = conv3x3(64, 256)
        # then up
        self.conv_up1 = conv3x3(128, 32)
        self.conv_up2 = conv3x3(32, 8)
        self.conv_up3 = conv3x3(8, 1)

        self.up1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.CIN1 = ConditionalInstanceNorm2dPlus(1, L)
        self.CIN2 = ConditionalInstanceNorm2dPlus(16, L)
        self.CIN3 = ConditionalInstanceNorm2dPlus(16, L)
        self.CIN4 = ConditionalInstanceNorm2dPlus(64, L)
        self.CIN5 = ConditionalInstanceNorm2dPlus(64, L)
        self.CIN6 = ConditionalInstanceNorm2dPlus(256, L)
        self.CIN7 = ConditionalInstanceNorm2dPlus(128, L)
        self.CIN8 = ConditionalInstanceNorm2dPlus(32, L)
        self.CIN9 = ConditionalInstanceNorm2dPlus(32, L)
        self.CIN10 = ConditionalInstanceNorm2dPlus(8, L)

        self.pool = nn.AvgPool2d(2, stride=2)
        self.activation = activation

        # we may use layer norm
        
    def forward(self, x, indices):
        bs = x.shape[0]
        # sigma = sigma.unsqueeze(1)
        assert x.shape == torch.Size([bs, 1, 28, 28])
        assert indices.shape == torch.Size([bs, ])

        # x = self.CIN1(x, indices)
        x = self.activation(self.conv1(x))
        res1 = x

        x = self.CIN2(x, indices)
        x = self.pool(x)
        # x = self.CIN3(x, indices)
        x = self.activation(self.conv2(x))
        res2 = x

        x = self.CIN4(x, indices)
        x = self.pool(x)
        # x = self.CIN5(x, indices)
        x = self.activation(self.conv3(x))

        # then up
        x = self.CIN6(x, indices)
        x = self.up1(x)
        x = torch.cat((x, res2), dim=1)
        x = self.CIN7(x, indices)
        x = self.activation(self.conv_up1(x))

        x = self.CIN8(x, indices)
        x = self.up2(x)
        x = torch.cat((x, res1), dim=1)
        x = self.CIN9(x, indices)
        x = self.activation(self.conv_up2(x))

        # x = self.CIN10(x, indices)
        x = self.conv_up3(x)

        assert x.size() == torch.Size([bs, 1, 28, 28])
        return x
    
class UNetv2(nn.Module):
    def __init__(self, sigmas: torch.Tensor, activation=F.relu):
        super(UNetv2, self).__init__()

        # the input shape is 1x28x28

        self.conv1 = conv3x3(1, 16)
        self.conv2 = conv3x3(16, 64)
        self.conv3 = conv3x3(64, 256)
        # then up
        self.conv_up1 = conv3x3(128, 32)
        self.conv_up2 = conv3x3(32, 8)
        self.conv_up3 = conv3x3(8, 1)

        self.up1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.pool = nn.AvgPool2d(2, stride=2)
        self.activation = activation

        self.sigmas = sigmas

        # we may use layer norm
        
    def forward(self, x, indices):
        bs = x.shape[0]
        # sigma = sigma.unsqueeze(1)
        assert x.shape == torch.Size([bs, 1, 28, 28])
        assert indices.shape == torch.Size([bs, ])

        x = self.activation(self.conv1(x))
        res1 = x

        x = self.pool(x)
        x = self.activation(self.conv2(x))
        res2 = x

        x = self.pool(x)
        x = self.activation(self.conv3(x))

        # then up
        x = self.up1(x)
        x = torch.cat((x, res2), dim=1)
        x = self.activation(self.conv_up1(x))

        x = self.up2(x)
        x = torch.cat((x, res1), dim=1)
        x = self.activation(self.conv_up2(x))

        x = self.conv_up3(x)

        assert x.size() == torch.Size([bs, 1, 28, 28])
        noise_level = self.sigmas[indices]
        x = x / noise_level
        return x