import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def plot_image_tensor(tensor, ax=None, show=False):
    image = transforms.ToPILImage()(tensor.cpu())

    if not ax:
        _, ax = plt.subplots()

    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if show:
        plt.show()
