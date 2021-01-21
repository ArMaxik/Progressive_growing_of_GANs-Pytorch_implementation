import torch
import numpy as np

import matplotlib.pyplot as plt

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