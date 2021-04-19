import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from lib.data import makeCatsDataset
from lib.networks import Progressive_Discriminator, Progressive_Generator

from lib.misc import noisy, image_with_title, prep_dirs, save_opt, remove_module_from_state_dict

import math
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import timedelta

from torchsummary import summary
class NeuralGenerator():
    def __init__(self, opt):
        self.latent = opt["latent"]
        self.isize = opt["isize"]
        self.batch = opt["batch"]
        self.cur_isize = 4
        self.device = opt["device"]
        self.size_feature_start_dec = opt["size_feature_start_dec"]
        self.gweights = opt["gweights"]

        self.gen = Progressive_Generator(self.latent)
        print("Setting up generator")

        while self.cur_isize < self.isize :
            if self.cur_isize < self.size_feature_start_dec // 2:
                div = 1
            else:
                div = 2
            self.gen.add_block(div=div)
            self.gen.end_transition()
            self.cur_isize *= 2
        
        print("Loading weights")
        weights = torch.load(self.gweights, map_location=self.device)
        self.gen.load_state_dict(weights)
        self.gen.to(self.device)

    def generate(self):
        latent = torch.randn(self.batch, self.latent, device=self.device)
        img = self.gen(latent).detach().cpu()
        return img

class Progressive_GAN(nn.Module):
    def __init__(self, opt):
        super(Progressive_GAN, self).__init__()
        self.exp_name = opt["exp_name"]
        self.batch = opt["batch"]
        self.latent = opt["latent"]
        self.isize = opt["isize"]
        self.cur_isize = 4
        self.size_feature_start_dec = opt["size_feature_start_dec"]
        self.device = opt["device"]
        self.device_ids = opt["device_ids"]
        self.data_path = opt["data_path"]

        self.epochs = opt["epochs"]
        self.lr_d = opt["lr_d"]
        self.lr_g = opt["lr_g"]
        self.g_it = opt["g_it"]
        self.d_it = opt["d_it"]
        self.b1 = opt["b1"]
        self.b2 = opt["b2"]
        self.noise = opt["noise"]
        self.lambda_coff = opt["lambda_coff"]
        self.eps_drift = opt["eps_drift"]

        self.gen = Progressive_Generator(self.latent)
        self.dis = Progressive_Discriminator()

        load_weights = self.cur_isize < opt["start_size"]
        while self.cur_isize < opt["start_size"]:
            if self.cur_isize < self.size_feature_start_dec // 2:
                div = 1
            else:
                div = 2
            self.gen.add_block(div=div)
            self.dis.add_block(div=div)
            self.gen.end_transition()
            self.dis.end_transition()
            self.cur_isize *= 2

        if load_weights:
            print("Loading generator weights")
            gweights = torch.load(opt["gweights"], map_location=self.device)
            self.gen.load_state_dict(gweights)
            self.gen.to(self.device)

            print("Loading discriminator weights")  
            dweights = torch.load(opt["dweights"], map_location=self.device)
            self.dis.load_state_dict(dweights)
            self.dis.to(self.device)

        if len(self.device_ids) == 0 :
            self.device_ids = [0]
        self.gen = nn.DataParallel(self.gen, device_ids=self.device_ids)
        self.dis = nn.DataParallel(self.dis, device_ids=self.device_ids)

        self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)

        prep_dirs(opt)
        save_opt(opt)

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
            if self.pbar.n % 10 == 0:
                end_time = time.time()
                tqdm.write(f"Size: {self.cur_isize}x{self.cur_isize} | Batch: {self.batch} | Transition: {self.transition} | G_loss: {self.g_loss.item()} D_loss: {self.d_loss.item()} | Total time: {timedelta(seconds=end_time - self.start_time)}")

        self.make_stats()
        self.make_chart()
        self.save_progress_image()

    def make_stats(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise, self.alpha).detach().cpu()
        vutils.save_image(
            fake, self.save_folder + f"/progress/img_{len(self.G_losses)}.png",
            padding=0, normalize=True, nrow=6
        )
        end_time = time.time()
        tqdm.write(f"Size: {self.cur_isize}x{self.cur_isize} | Batch: {self.batch} | Transition: {self.transition} | Min: {fake.min()} Max: {fake.max()} | Total time: {timedelta(seconds=end_time - self.start_time)}")

        self.G_losses.append(self.g_loss.item())
        self.D_losses.append(self.d_loss.item())

    def train(self):
        self.save_folder = os.path.join('./out', self.exp_name + '/')
        self.start_time = time.time()

        print("Strated {}\nepochs: {}\ndevice: {}".format(self.exp_name, self.epochs, self.device))
        
        self.setup_train()

        self.pbar = tqdm(bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')


        while self.cur_isize < self.isize:
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 

            self.transition = False
            self.alpha = -1  # No transition
            self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
            if self.cur_isize != 128:
                for epoch in range(self.epochs):
                    self.train_one_epoch()
                    self.save_progress_image()
            
            self.save_weights()
            self.transition = True
            if self.cur_isize < self.size_feature_start_dec // 2:
                div = 1
            else:
                div = 2

            self.gen.module.add_block(div=div)
            self.dis.module.add_block(div=div)

            self.gen.module.to(self.device)
            self.dis.module.to(self.device)
            
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 

            # self.epochs = int(self.epochs*1.15)
            alpha_inc = 1.0 / (self.epochs + 1)

            self.cur_isize *= 2

            if self.cur_isize == 256:
                self.batch = 14 * len(self.device_ids)
            if self.cur_isize == 512:
                self.batch = 6 * len(self.device_ids)
                # self.data_path = "/raid/veliseev/datasets/cats/cats_faces_hd/512"
            if self.cur_isize == 1024:
                self.batch = 3 * len(self.device_ids)

            self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)
            self.alpha = alpha_inc

            self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
            for epoch in range(self.epochs):
                self.train_one_epoch()
                self.save_progress_image()
                self.alpha += alpha_inc
            
            self.gen.module.end_transition()
            self.dis.module.end_transition()

            # self.epochs = int(self.epochs*1.15)
            self.save_weights()

        self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`
        self.transition = False
        self.alpha = -1  # No transition
        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.save_progress_image()

        self.save_weights()

    def gradien_penalty(self, imgs_real, imgs_fake):
        b, c, h, w = imgs_real.shape
        epsilon = torch.rand((b, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolate = epsilon*imgs_real + (1.0 - epsilon)*imgs_fake
        interpolate.requires_grad_(True)

        d_interpolate = self.dis(interpolate, self.alpha).view(-1)

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

        output_real = self.dis(imgs_real, self.alpha).view(-1)

        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)
        imgs_fake = self.gen(z, self.alpha)

        output_fake = self.dis(imgs_fake, self.alpha).view(-1)

        self.d_loss = output_fake.mean() - output_real.mean() + self.gradien_penalty(imgs_real, imgs_fake)
        self.d_loss += self.eps_drift * torch.mean(output_real ** 2)
        self.d_loss.backward()

        for name, param in self.dis.named_parameters():
            i_inf = torch.isfinite(param.grad).all()
            if not i_inf:
                tqdm.write(f"{name} {i_inf}")

        self.D_G_z1 = output_fake.mean().item()

        self.op_dis.step()

    def train_generator(self):
        self.op_gen.zero_grad()

        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)
        imgs = self.gen(z, self.alpha)
            
        if self.noise:
            imgs = noisy(imgs, self.device)
        output_fake = self.dis(imgs, self.alpha).view(-1)
        
        self.g_loss = -output_fake.mean()
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()

    def save_progress_image(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise_64, self.alpha).detach().cpu()

        name = f"final_{self.cur_isize}x{self.cur_isize}"
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

    def save_weights(self):
        g_w = self.gen.state_dict()
        d_w = self.dis.state_dict()
        remove_module_from_state_dict(g_w)
        remove_module_from_state_dict(d_w)

        postfix = f"{self.cur_isize}"
        if self.transition:
            postfix += "_tr"
        torch.save(g_w, self.save_folder + f'c_gen_{postfix}.pth')
        torch.save(d_w, self.save_folder + f'c_dis_{postfix}.pth')
