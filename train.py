from lib.model import Progressive_GAN
from lib.misc import make_video

from datetime import timedelta
import argparse
import torch
import json
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
parser.add_argument("-d", "--device", help="using device", metavar ="Device", choices=["cpu", "cuda"], type=str, default="cpu")
parser.add_argument("--dev_ids", help="device ids", metavar ="IDs", nargs='+', type=int, default=[0])
parser.add_argument("--data", help="dataset path", metavar ="Data", type=str, default="/raid/veliseev/datasets/cats/faces_1024_jpg/")

args = parser.parse_args()
with open(args.config, 'r') as config:
    config_j = json.loads(config.read())
    
config_j["device"] = args.device
config_j["device_ids"] = args.dev_ids
config_j["data_path"] = args.data

if config_j["device"] == "cuda" and torch.cuda.is_available():
    config_j["device"] = torch.device(f"cuda:{config_j['device_ids'][0]}")
else:
    config_j["device"] = torch.device("cpu")

start_time = time.time()

gan = Progressive_GAN(config_j)
gan.train()

print("Making video")
make_video(config_j)

end_time = time.time()
print(f"Total {timedelta(seconds=end_time - start_time)}\n")
