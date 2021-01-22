from lib.model import Progressive_GAN
from lib.misc import make_video

from datetime import timedelta
import torch
import time
import os

class options:
    def __init__(self):
        self.exp_name = "Progressive_GAN_02"
        self.batch = 64
        self.latent = 512
        self.isize = 128
        self.device_ids = [2, 3]
        self.device = torch.device(f"cuda:{self.device_ids[0]}" if (torch.cuda.is_available()) else "cpu")
        # self.data_path = "/home/v-eliseev/Datasets/cats/"
        # self.data_path = "/mnt/p/datasets/cats/"
        self.data_path = "/raid/veliseev/datasets/cats/imgs/"

        self.epochs = 30
        self.lr_d = 0.0001
        self.lr_g = 0.0001
        self.lr_decay_epoch = []
        self.lr_decay_factor = 10.0
        self.g_it = 1
        self.d_it = 1
        self.b1 = 0.0
        self.b2 = 0.99
        self.noise = False
        self.lambda_coff = 10.0
    

opt = options()
if not os.path.isdir(f"./out/{opt.exp_name}"):
    os.makedirs(f"./out/{opt.exp_name}")
if not os.path.isdir(f"./out/{opt.exp_name}/progress"):
    os.makedirs(f"./out/{opt.exp_name}/progress")
with open(f"./out/{opt.exp_name}/opt.txt", 'w') as f:
    for (k, v) in opt.__dict__.items():
        f.write("{:24s}{}\n".format(k, v))

start_time = time.time()

gan = Progressive_GAN(opt)
gan.train()

print("Making video")
make_video(opt)

end_time = time.time()
print(f"Total {timedelta(seconds=end_time - start_time)}")