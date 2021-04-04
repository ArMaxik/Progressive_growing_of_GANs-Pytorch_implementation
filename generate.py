from lib.model import NeuralGenerator

import torchvision.utils as vutils
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output directory", metavar="Output", type=str, required=True)
parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
parser.add_argument("-n", "--number", help="number of generated images", metavar ="Number", type=int, default=1)
parser.add_argument("-d", "--device", help="using device", metavar ="Device", type=str, default="cpu")

args = parser.parse_args()

with open(args.config, 'r') as config:
    config_j = json.loads(config.read())

config_j["device"] = args.device
print(args)

gen = NeuralGenerator(config_j)
for b in range(args.number // config_j["batch"]+1):
    img = gen.generate()
    i = 0
    while i < config_j["batch"] and i < args.number - b*config_j["batch"]:
        num = i+b*config_j["batch"]
        print(f"Generating image: {num+1:{len(str(args.number))}}/{args.number}")
        vutils.save_image(
                img[i],
                os.path.join(args.output + f"generated_cat_{num}.png"),
                padding=0,
                nrow=1,
                normalize=True,
            )
        i += 1
