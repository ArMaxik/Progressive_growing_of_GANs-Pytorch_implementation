from lib.model import Progressive_GAN
from lib.misc import make_video

from datetime import timedelta
import torch
import time
import os

class options:
    def __init__(self):
        self.exp_name = "Progressive_GAN_09"
        self.batch = 32
        self.latent = 512
        self.isize = 512
        self.size_feature_start_dec = 64
        self.device_ids = [1, 2]
        self.device = torch.device(f"cuda:{self.device_ids[0]}" if (torch.cuda.is_available()) else "cpu")
        # self.data_path = "/home/v-eliseev/Datasets/cats/"
        # self.data_path = "/mnt/p/datasets/cats/"
        # self.data_path = "/raid/veliseev/datasets/cats/imgs/"
        self.data_path = "/raid/veliseev/datasets/cats/faces_1024_jpg/"

        self.epochs = 60
        self.lr_d = 0.0004
        self.lr_g = 0.0004
        self.eps_drift = 0.001
        self.g_it = 1
        self.d_it = 1
        self.b1 = 0.0
        self.b2 = 0.99
        self.noise = False
        self.lambda_coff = 10.0
    
# 7
opt = options()

start_time = time.time()

gan = Progressive_GAN(opt)
gan.train()

print("Making video")
make_video(opt)

end_time = time.time()
print(f"Total {timedelta(seconds=end_time - start_time)}\n")
