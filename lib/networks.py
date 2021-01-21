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
        self.eps = 1e-8

    def forward(self, x):
        std = x.std(dim=0, unbiased=False).mean()
        std = std.expand((x.shape[0], 1, x.shape[2], x.shape[3]))

        x = torch.cat((x, std), dim=1)

        return x

class Progressive_Generator(nn.Module):
    def __init__(self, LATENT, device="cpu", device_ids=[]):
        super(Progressive_Generator, self).__init__()
        self.to(device)
        self.device = device
        self.device_ids = device_ids
        
        self.z = LATENT
        
        self.nc = 512
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(self.z, self.nc, kernel_size=4, stride=1, padding=0, bias=True).to(self.device),
            PixelNorm().to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
            nn.Conv2d(self.nc, self.nc, 3, stride=1, padding=1, bias=True).to(self.device),
            PixelNorm().to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
        ])
        self.toRGB = nn.Conv2d(self.nc, 3, (1, 1), bias=True).to(self.device)

        # self.block_size = 4

        for l in self.layers:
            weights_init(l)
        weights_init(self.toRGB)
        
        if len(self.device_ids) > 1 and not (self.device == "cpu"):
            self.layers = nn.ModuleList([nn.DataParallel(l, device_ids=self.device_ids) for l in self.layers])
            self.toRGB = nn.DataParallel(self.toRGB, device_ids=self.device_ids)

        
    def add_block(self):
        block = nn.ModuleList([
            nn.Upsample(scale_factor=2.0).to(self.device),
            nn.Conv2d(self.nc, self.nc // 2, kernel_size=3, stride=1, padding=1, bias=True).to(self.device),
            PixelNorm().to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
            nn.ConvTranspose2d(self.nc // 2, self.nc // 2, (3, 3), stride=1, padding=1, bias=True).to(self.device),
            PixelNorm().to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
        ])
        self.block_size = len(block)

        self.toRGB_new = nn.Conv2d(self.nc // 2, 3, (1, 1), bias=True).to(self.device)

        for l in block:
            weights_init(l)
        weights_init(self.toRGB_new)

        if len(self.device_ids) > 1 and not (self.device == "cpu"):
            block = nn.ModuleList([nn.DataParallel(l, device_ids=self.device_ids) for l in block])
            self.toRGB_new = nn.DataParallel(self.toRGB_new, device_ids=self.device_ids)

        self.layers.extend(block)
        self.nc //= 2
            

    def forward(self, x):
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
    def __init__(self, device="cpu", device_ids=[]):
        super(Progressive_Discriminator, self).__init__()

        self.device = device
        self.device_ids = device_ids
        
        self.nc = 512
        self.layers = nn.ModuleList([
            MinibatchStd().to(self.device),
            nn.Conv2d(self.nc+1, self.nc, kernel_size=3, stride=1, padding=1, bias=True).to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
            nn.Conv2d(self.nc, self.nc, kernel_size=4, stride=1, padding=0, bias=True).to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
        ])
        self.fromRGB = nn.Conv2d(3, self.nc, (1, 1), bias=True).to(self.device)
        self.lrelu_fromRGB = nn.LeakyReLU(0.2).to(self.device)

        # self.block_size = 3

        self.linear = nn.Linear(in_features = self.nc, out_features = 1).to(self.device)

        for l in self.layers:
            weights_init(l)
        weights_init(self.fromRGB)
        weights_init(self.linear)

        if len(self.device_ids) > 1 and not (self.device == "cpu"):
            self.layers = nn.ModuleList([nn.DataParallel(l, device_ids=self.device_ids) for l in self.layers])
            self.fromRGB = nn.DataParallel(self.fromRGB, device_ids=self.device_ids)
            self.lrelu_fromRGB = nn.DataParallel(self.lrelu_fromRGB, device_ids=self.device_ids)
            self.linear = nn.DataParallel(self.linear, device_ids=self.device_ids)


    def add_block(self):
        block = nn.ModuleList([
            nn.Conv2d(self.nc//2, self.nc//2, kernel_size=3, stride=1, padding=1, bias=True).to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
            nn.Conv2d(self.nc//2, self.nc, kernel_size=3, stride=1, padding=1, bias=True).to(self.device),
            nn.LeakyReLU(0.2).to(self.device),
            nn.AvgPool2d(2).to(self.device),
        ])
        self.block_size = len(block)
        self.fromRGB_new = nn.Conv2d(3, self.nc // 2, (1, 1), bias=True).to(self.device)
        # self.inorm_fromRGB_new = nn.InstanceNorm2d(self.nc // 2).to(self.device)
        
        for l in block:
            weights_init(l)
        weights_init(self.fromRGB_new)
        # weights_init(self.inorm_fromRGB_new)

        if len(self.device_ids) > 1 and not (self.device == "cpu"):
            block = nn.ModuleList([nn.DataParallel(l, device_ids=self.device_ids) for l in block])
            self.fromRGB_new = nn.DataParallel(self.fromRGB_new, device_ids=self.device_ids)
            # self.inorm_fromRGB_new = nn.DataParallel(self.inorm_fromRGB_new, device_ids=self.device_ids)

        
        self.layers = block.extend(self.layers)
        self.nc //= 2


    def forward(self, x):
        x = self.fromRGB(x)
        x = self.lrelu_fromRGB(x)
        # x = self.inorm_fromRGB(x)

        for layer in self.layers:
            x = layer(x)

        x = self.linear(x.view(-1, x.shape[1]))
        return x

    def transition_forward(self, x, alpha):
        # x_old = nn.functional.interpolate(x, size = x.shape[2] // 2)
        x_old = torch.nn.functional.avg_pool2d(x, kernel_size = 2)
        x_old = self.fromRGB(x_old)
        x_old = self.lrelu_fromRGB(x_old)
        # x_old = self.inorm_fromRGB(x_old)

        x_new = self.fromRGB_new(x)
        x_new = self.lrelu_fromRGB(x_new)
        # x_new = self.inorm_fromRGB_new(x_new)
        for layer in self.layers[:self.block_size]: 
            x_new = layer(x_new)
        
        x = x_new * alpha + x_old * (1.0 - alpha)

        for layer in self.layers[self.block_size:]:
            x = layer(x)

        x = self.linear(x.view(-1, x.shape[1]))
        return x
    
    def end_transition(self):
        self.fromRGB = self.fromRGB_new
        # self.inorm_fromRGB = self.inorm_fromRGB_new


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
