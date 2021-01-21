import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import data
from torchsummary import summary
from networks import *
from misc import *

# DATA_PATH = "/home/v-eliseev/Datasets/cats/"
DATA_PATH = "/raid/veliseev/datasets/cats/"
# DATA_PATH = "/mnt/p/datasets/cats/"

def imshow(img, name=None):
    fig, ax = plt.subplots()
    img = np.transpose(img.numpy(), (1, 2, 0))
    ax.imshow(img, interpolation='none')
    ax.axis('off')

    if name != None:
        fig.tight_layout()
        fig.savefig(name + ".png")
    else:
        fig.show()
    plt.close()

def image_with_title(img, title_text, info_text):
    plt.axis('off')
    title = plt.text(0,-7,
                    title_text, 
                    fontsize=26)
    title.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    info = plt.text(0,64*6+22,
                    info_text, 
                    fontsize=14)
    info.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    img_n = plt.imshow(np.transpose(img,(1,2,0)), animated=True)
    return [img_n, title]

dataloader = data.makeCatsDataset(path=DATA_PATH, batch=16, isize=64)
print(f"DATA lenght {len(dataloader)}")
img_list = []
for i_batch, im in enumerate(dataloader):
    im = noisy(im)
    im = (im+1.0)/2.0
    
    # imshow(torchvision.utils.make_grid(im, nrow=4), name=str(i_batch))
    img_list.append(torchvision.utils.make_grid(im, nrow=4))
    if i_batch == 100:
        break

fig = plt.figure(figsize=(12,12))
# fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
ims = [
    image_with_title(img,
                     "Epoch: {}".format(i),
                     "[RGAN] Batch size: {0}, Latent space: {1}, size {2}x{2}".format(16, 15, 32))
    for i, img in enumerate(img_list)
    ]
ani = animation.ArtistAnimation(fig, ims, interval=1500, repeat_delay=1000, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=5000, codec='mpeg4')
# ani.save('test.mp4', writer=writer)


LATENT = 100
print("== GAN testing")
gen = Progressive_Generator(LATENT, device="cpu", device_ids=[1])
gen.add_block()
gen.end_transition()
gen.add_block()
gen.end_transition()
gen.add_block()

dis = Progressive_Discriminator(device="cpu", device_ids=[1])
dis.add_block()
dis.end_transition()
dis.add_block()
dis.end_transition()
dis.add_block()


# print(*gen.layers, sep='\n')
# print(gen.toRGB)
# print(gen.toRGB_new)

# print()
# print(dis.fromRGB)
# print(dis.fromRGB_new)
# print(*dis.layers, sep='\n')


print("== Transition testing")
noise = torch.randn(16, LATENT).to("cpu")
data = gen.transition_forward(noise, 0.2)
dis.transition_forward(data.to("cpu"), 0.2)

print("== Normal testing")
gen.end_transition()
dis.end_transition()
data = gen(noise).cpu()

torchvision.utils.save_image(data, "test.png", nrow=4, normalize=True)

summary(gen, (3, LATENT), device="cpu")
summary(dis, (3, 32, 32), device="cpu")

