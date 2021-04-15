import torchvision
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import data
from torchsummary import summary
from networks import *
from misc import *

# DATA_PATH = "/home/v-eliseev/Datasets/cats/"
DATA_PATH = "/raid/veliseev/datasets/cats/cats_faces_hd/512"
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

dataloader = data.makeCatsDataset(path=DATA_PATH, batch=16, isize=512)
print(f"DATA lenght {len(dataloader)}")
img_list = []
for i_batch, (im, _) in enumerate(dataloader):
    im = noisy(im)
    im = (im+1.0)/2.0
    
    # imshow(torchvision.utils.make_grid(im, nrow=4), name=str(i_batch))
    vutils.save_image(
            im, f"./img_{i_batch}.png",
            padding=5, normalize=True, nrow=4
        )
    img_n = torchvision.utils.make_grid(im, nrow=4).numpy()
    img_list.append(img_n)
    if i_batch == 5:
        break
print(img_n.shape, img_n.dtype, np.min(img_n), np.max(img_n))
fig = plt.figure(figsize=(12,12))
# fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
ims = [
    image_with_title(img,
                     "Epoch: {}".format(i),
                     "[RGAN] Batch size: {0}, Latent space: {1}, size {2}x{2}".format(16, 15, 32))
    for i, img in enumerate(img_list)
    ]
# ani = animation.ArtistAnimation(fig, ims, interval=1500, repeat_delay=1000, blit=True)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=5000, codec='mpeg4')
# ani.save('test.mp4', writer=writer)


LATENT = 512
print("== GAN testing")
gen = Progressive_Generator(LATENT)
gen.add_block(div=1)
gen.end_transition()
gen.add_block(div=1)
gen.end_transition()
gen.add_block(div=1)
gen.end_transition()
gen.add_block()
gen.end_transition()
gen.add_block()
gen.end_transition()
gen.add_block()
gen.end_transition()
gen.add_block()

dis = Progressive_Discriminator()
dis.add_block(div=1)
dis.end_transition()
dis.add_block(div=1)
dis.end_transition()
dis.add_block(div=1)
dis.end_transition()
dis.add_block()
dis.end_transition()
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
summary(dis, (3, 512, 512), device="cpu")

