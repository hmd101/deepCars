import matplotlib.pyplot as plt
import torchvision
import numpy as np
from data.car_dataset import inverse_transform


def imshow(img, ax=None):
    img = inverse_transform(img)

    img = img.to("cpu")
    img = torchvision.utils.make_grid(img, nrow=4)
    npimg = img.numpy()
    # rotate image by transposing dimensions accordingly
    npimg = np.transpose(npimg, (1, 2, 0))

    if ax is None:
        plt_img = plt.imshow(npimg)
    else:
        plt_img = ax.imshow(npimg)

    # Hide X and Y axes label marks
    plt_img.axes.xaxis.set_tick_params(labelbottom=False)
    plt_img.axes.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    plt_img.axes.set_xticks([])
    plt_img.axes.set_yticks([])

    return plt_img
