import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class MinibatchStd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = x.std(dim=0, unbiased=False).mean()
        std = std.expand((x.shape[0], 1, x.shape[2], x.shape[3]))

        x = torch.cat((x, std), dim=1)

        return x

class EqualConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EqualConv2d, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5
        # nn.init.xavier_normal_(self.weight.data)
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x
        # weight_n = self.weight.data / self.c

        # return super()._conv_forward(x, weight_n)

class EqualConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(EqualConvTranspose2d, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5
        # nn.init.xavier_normal_(self.weight.data)
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x
    #     if self.padding_mode != 'zeros':
    #         raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

    #     output_padding = self._output_padding(
    #         x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

    #     weight_n = self.weight.data / self.c
    #     return F.conv_transpose2d(
    #         x, weight_n, self.bias, self.stride, self.padding,
    #         output_padding, self.groups, self.dilation)

class EqualLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(EqualLinear, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5
        # nn.init.xavier_normal_(self.weight.data)
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x
    #     weight_n = self.weight.data / self.c
    #     return F.linear(x, weight_n, self.bias)


class Progressive_Generator(nn.Module):
    def __init__(self, LATENT):
        super(Progressive_Generator, self).__init__()
        
        self.z = LATENT
        
        self.nc = 512
        self.layers = nn.ModuleList([
            EqualConvTranspose2d(self.z, self.nc, kernel_size=4, stride=1, padding=0, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.nc, self.nc, 3, stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ])
        self.toRGB = EqualConv2d(self.nc, 3, (1, 1), bias=True)

        # self.block_size = 4

    def add_block(self):
        block = nn.ModuleList([
            nn.Upsample(scale_factor=2.0),
            EqualConv2d(self.nc, self.nc // 2, kernel_size=3, stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.nc // 2, self.nc // 2, (3, 3), stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ])
        self.block_size = len(block)

        self.toRGB_new = EqualConv2d(self.nc // 2, 3, (1, 1), bias=True)

        self.layers.extend(block)
        self.nc //= 2
            

    def forward(self, x, alpha = -1):
        if not alpha == -1:
            return self.transition_forward(x, alpha)
        
        return self.normal_forward(x) 

    def normal_forward(self, x):
        x = x.view(-1, self.z, 1, 1)
        for layer in self.layers:
            x = layer(x)

        x = self.toRGB(x)
        x = torch.tanh(x)
        return x

    def transition_forward(self, x, alpha):
        x = x.view(-1, self.z, 1, 1)
        for layer in self.layers[:-self.block_size]:
            x = layer(x)

        x_old = nn.functional.interpolate(x, size = x.shape[2] * 2)
        x_old = self.toRGB(x_old)
        x_old = torch.tanh(x_old)

        x_new = x
        for layer in self.layers[-self.block_size:]:
            x_new = layer(x_new)
        x_new = self.toRGB_new(x_new)
        x_new = torch.tanh(x_new)

        x = x_new * alpha + x_old * (1.0 - alpha)
        return x

    def end_transition(self):
        self.toRGB = self.toRGB_new

class Progressive_Discriminator(nn.Module):
    def __init__(self):
        super(Progressive_Discriminator, self).__init__()
        
        self.nc = 512
        self.layers = nn.ModuleList([
            MinibatchStd(),
            EqualConv2d(self.nc+1, self.nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.nc, self.nc, kernel_size=4, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2),
        ])
        self.fromRGB = EqualConv2d(3, self.nc, (1, 1), bias=True)
        self.lrelu_fromRGB = nn.LeakyReLU(0.2)

        # self.block_size = 3

        self.linear = EqualLinear(in_features = self.nc, out_features = 1)

    def add_block(self):
        block = nn.ModuleList([
            EqualConv2d(self.nc//2, self.nc//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.nc//2, self.nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        ])
        self.block_size = len(block)
        self.fromRGB_new = EqualConv2d(3, self.nc // 2, (1, 1), bias=True)
        
        self.layers = block.extend(self.layers)
        self.nc //= 2


    def forward(self, x, alpha = -1):
        if not alpha == -1:
            return self.transition_forward(x, alpha)
        
        return self.normal_forward(x) 

    def normal_forward(self, x):
        x = self.fromRGB(x)
        x = self.lrelu_fromRGB(x)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(x.view(-1, x.shape[1]))
        return x

    def transition_forward(self, x, alpha):
        x_old = torch.nn.functional.avg_pool2d(x, kernel_size = 2)
        x_old = self.fromRGB(x_old)
        x_old = self.lrelu_fromRGB(x_old)

        x_new = self.fromRGB_new(x)
        x_new = self.lrelu_fromRGB(x_new)
        for layer in self.layers[:self.block_size]: 
            x_new = layer(x_new)
        
        x = x_new * alpha + x_old * (1.0 - alpha)

        for layer in self.layers[self.block_size:]:
            x = layer(x)

        x = self.linear(x.view(-1, x.shape[1]))
        return x
    
    def end_transition(self):
        self.fromRGB = self.fromRGB_new

