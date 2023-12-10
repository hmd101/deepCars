import matplotlib.pyplot as plt
import torchvision
import numpy as np
from data.car_dataset import inverse_transform


def imshow(img):
    img = inverse_transform(img)

    img = img.to("cpu")
    img = torchvision.utils.make_grid(img, nrow=4)
    npimg = img.numpy()
    # rotate image by transposing dimensions accordingly
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()