from lib.model import NeuralGenerator

import torchvision.utils as vutils
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output directory", metavar="Output", type=str, required=True)
parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
parser.add_argument("-n", "--number", help="number of generated pictures", metavar ="Number", type=int, default=1)
parser.add_argument("-d", "--device", help="using device", metavar ="Device", choices=["cpu", "cuda"], type=str, default="cpu")

args = parser.parse_args()

with open(args.config, 'r') as config:
    config_j = json.loads(config.read())

config_j["device"] = args.device


gen = NeuralGenerator(config_j)
for i in range(args.number):
    print(f"Generating image: {i+1:{len(str(args.number))}}/{args.number}")
    img = gen.generate()
    vutils.save_image(
            img,
            args.output + f"/generated_cat_{i}.png",
            padding=0,
            normalize=True,
        )
