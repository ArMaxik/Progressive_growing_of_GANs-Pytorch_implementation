import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from lib.data import makeCatsDataset
from lib.networks import Progressive_Discriminator, Progressive_Generator, weights_init

from lib.misc import noisy, image_with_title

import math
import numpy as np
from tqdm import tqdm
import os

class Progressive_GAN(nn.Module):
    def __init__(self, opt):
        super(Progressive_GAN, self).__init__()
        self.exp_name = opt.exp_name
        self.batch = opt.batch
        self.latent = opt.latent
        self.isize = opt.isize
        self.cur_isize = 4
        self.device = opt.device
        self.device_ids = opt.device_ids
        self.data_path = opt.data_path

        self.epochs = opt.epochs
        self.lr_d = opt.lr_d
        self.lr_g = opt.lr_g
        self.lr_decay_epoch = opt.lr_decay_epoch
        self.lr_decay_factor = opt.lr_decay_factor
        self.g_it = opt.g_it
        self.d_it = opt.d_it
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.noise = opt.noise
        self.lambda_coff = opt.lambda_coff
        self.eps_drift = 0.001
        

        self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)
        self.gen = Progressive_Generator(self.latent, device=self.device, device_ids=self.device_ids)
        self.dis = Progressive_Discriminator(device=self.device, device_ids=self.device_ids)

    def setup_train(self):
        self.fixed_noise = torch.randn(36, self.latent, device=self.device)
        self.fixed_noise_64 = torch.randn(64, self.latent, device=self.device)
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.real_label = 0.9
        self.fake_label = 0.0

        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()

    def train_one_epoch(self):
        for i, (data, _) in enumerate(self.dataloader, 0):
            self.data_device = data.to(self.device)
            if self.noise:
                self.data_device = noisy(self.data_device, self.device)

            mini_it = i % (self.g_it + self.d_it-1)
            if mini_it < self.d_it:
                self.train_discriminator()
            if mini_it >= self.d_it-1:
                self.train_generator()

            self.pbar.update()
        self.make_stats()
        self.make_chart()
        self.save_progress_image()

    def make_stats(self):
        # with torch.no_grad():
            # fake = self.gen(self.fixed_noise).detach().cpu()
        # self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=6))

        self.G_losses.append(self.g_loss.item())
        self.D_losses.append(self.d_loss.item())

    def train(self):
        self.save_folder = os.path.join('./out', self.exp_name + '/')
        self.init_folder()

        print("Strated {}\nepochs: {}\ndevice: {}".format(self.exp_name, self.epochs, self.device))
        
        self.setup_train()

        self.pbar = tqdm()


        while self.cur_isize < self.isize:
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 

            print(f"train {self.cur_isize}x{self.cur_isize}")
            self.transition = False
            self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
            for epoch in range(self.epochs):
                self.train_one_epoch()
                self.save_progress_image()
            
            self.transition = True
            self.gen.add_block()
            self.dis.add_block()
            
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 

            self.epochs = int(self.epochs*1.15)
            alpha_inc = 1.0 / (self.epochs-1)

            self.cur_isize *= 2
            self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)
            self.alpha = alpha_inc

            print(f"train transition {self.cur_isize}x{self.cur_isize}")
            self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
            for epoch in range(self.epochs):
                self.train_one_epoch()
                self.save_progress_image()
                self.alpha += alpha_inc
            
            self.gen.end_transition()
            self.dis.end_transition()

            self.epochs = int(self.epochs*1.15)
            self.make_chart()
        print("train {}x{}".format(self.cur_isize, self.cur_isize))
        self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
        self.transition = False
        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.save_progress_image()

        self.save_weights()

        # self.save_video()

    def gradien_penalty(self, imgs_real, imgs_fake):
        b, c, h, w = imgs_real.shape
        epsilon = torch.rand((b, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolate = epsilon*imgs_real + (1.0 - epsilon)*imgs_fake
        interpolate.requires_grad_(True)

        if self.transition:
            d_interpolate = self.dis.transition_forward(interpolate, self.alpha).view(-1)
        else:
            d_interpolate = self.dis(interpolate).view(-1)

        gradients = torch.autograd.grad(
            outputs=d_interpolate,
            inputs=interpolate,
            grad_outputs=torch.ones(d_interpolate.shape, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(b, -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_coff*penalty

    def train_discriminator(self):
        self.op_dis.zero_grad()
        # True
        imgs_real = self.data_device

        if self.transition:
            output_real = self.dis.transition_forward(imgs_real, self.alpha).view(-1)
        else:
            output_real = self.dis(imgs_real).view(-1)

        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)
        if self.transition:
            imgs_fake = self.gen.transition_forward(z, self.alpha)
        else:
            imgs_fake = self.gen(z)

        if self.transition:
            output_fake = self.dis.transition_forward(imgs_fake, self.alpha).view(-1)
        else:
            output_fake = self.dis(imgs_fake).view(-1)

        self.d_loss = output_fake.mean() - output_real.mean() + self.gradien_penalty(imgs_real, imgs_fake)
        self.d_loss += self.eps_drift * torch.mean(output_real ** 2)
        self.d_loss.backward()

        self.D_G_z1 = output_fake.mean().item()

        self.op_dis.step()

    def train_generator(self):
        self.op_gen.zero_grad()

        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)

        if self.transition:
            imgs = self.gen.transition_forward(z, self.alpha)
        else:
            imgs = self.gen(z)
            
        if self.noise:
            imgs = noisy(imgs, self.device)

        if self.transition:
            output_fake = self.dis.transition_forward(imgs, self.alpha).view(-1)
        else:
            output_fake = self.dis(imgs).view(-1)
        
        self.g_loss = -output_fake.mean()
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()

    def save_progress_image(self):
        with torch.no_grad():
            if self.transition:
                fake = self.gen.transition_forward(self.fixed_noise_64, self.alpha).detach().cpu()
            else:
                fake = self.gen(self.fixed_noise_64).detach().cpu()
        name = "final_{}x{}".format(self.cur_isize, self.cur_isize)
        if self.transition:
            name += "_transition"
        name += ".png"

        vutils.save_image(fake, self.save_folder + name, normalize=True)

    def make_chart(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_folder + "losses.png")
        plt.close()

        # noise = torch.randn(64, self.latent, device=self.device)
        # with torch.no_grad():
        #     fake = self.gen(noise).detach().cpu()
        # vutils.save_image(fake, self.save_folder + "final.png", normalize=True)

    def save_video(self):
        fig = plt.figure(figsize=(12,12))
        ims = [
            image_with_title(img,
                            "Epoch: {}".format(i),
                            "[WGAN] batch size: {0}, latent space: {1}, size {2}x{2}".format(self.batch, self.latent, 32))
            for i, img in enumerate(self.img_list)
            ]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=3500, codec='mpeg4')
        # ani.save(self.save_folder + self.exp_name +'_hist.mp4', writer=writer)

    def save_weights(self):
        g_w = self.gen.state_dict()
        d_w = self.dis.state_dict()
        torch.save(g_w, self.save_folder + 'c_gen.pth')
        torch.save(d_w, self.save_folder + 'c_dis.pth')

    def init_folder(self):
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
