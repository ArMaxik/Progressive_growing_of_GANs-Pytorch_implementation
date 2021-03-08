import torch
import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os

def noisy(image, device='cpu:0'):
    b = -0.11
    a = 0.11
    mean = np.random.random_sample()*(b - a) + a
    var = 0.015
    sigma = var**0.5
    gauss = torch.randn(image.size(), device=device)*sigma  + mean
    noisy = image + gauss
    return torch.clamp(noisy, -1.0, 1.0)

def image_with_title(img, title_text, info_text):
    plt.axis('off')
    title = plt.text(0,-12,
                    title_text, 
                    fontsize=26)
    title.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    # info = plt.text(0,32*6+22,
    #                 info_text, 
    #                 fontsize=14)
    # info.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    img_n = plt.imshow(np.transpose(img,(1,2,0)), animated=True)
    return [img_n, title]

def make_video(opt):
    i = 0
    imgs = []
    size = (6*opt["isize"], 6*opt["isize"])
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    while True:
        img = cv2.imread(f"./out/{opt['exp_name']}/progress/img_{i}.png")
        if img is None:
            break
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        plt.axis('off')

        imgs.append([plt.imshow(img[:,:,::-1], animated=True)])

        i += 1

    ani = animation.ArtistAnimation(fig, imgs, interval=600, repeat_delay=600, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=7500, codec='mpeg4')
    ani.save(f"./out/{opt['exp_name']}/{opt['exp_name']}" +'_hist.mp4', writer=writer)

def prep_dirs(opt):
    if not os.path.isdir(f"./out/{opt['exp_name']}"):
        os.makedirs(f"./out/{opt['exp_name']}")
    if not os.path.isdir(f"./out/{opt['exp_name']}/progress"):
        os.makedirs(f"./out/{opt['exp_name']}/progress")

def save_opt(opt):
    with open(f"./out/{opt['exp_name']}/opt.txt", 'w') as f:
        for (k, v) in opt.items():
            f.write("{:24s}{}\n".format(k, v))

def remove_module_from_state_dict(state_dict):
    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = key.replace('module.', '')
        state_dict[new_key] = state_dict[key]
        state_dict.pop(key)