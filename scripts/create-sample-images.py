import os

import matplotlib.pyplot as plt
import torch

from csfnst.utils import load_image, plot_image_tensor, get_checkpoint
import torchvision.transforms as transforms

flatten = lambda l: [item for sublist in l for item in sublist]

image = load_image('./images/content/htw-768x768.jpg').unsqueeze(0)
output_image_path = './images/output/samples/'
checkpoints_path = './checkpoints/'
checkpoints = [
    checkpoint
    for checkpoint in os.listdir(checkpoints_path)
    if checkpoint != '.gitkeep'
]
checkpoints.sort()

fig, axs = plt.subplots(4, 4, figsize=(8, 8), sharex='row', sharey='col')
fig.subplots_adjust(wspace=.025, hspace=.025)
axs = flatten(axs)

for i, checkpoint in enumerate(checkpoints):
    checkpoint, model = get_checkpoint(f'{checkpoints_path}{checkpoint}')
    # name = splitext(checkpoint)[0]
    output_tensor = model(image).squeeze()
    # save_image_tensor(output_image, f'{output_image_path}{name}.jpg')
    # output_tensor = torch.randn((3, 768, 768))
    output_image = transforms.ToPILImage()(output_tensor.cpu())

    # axs[i].set_title(f'{i + 1}')
    if i % 4 == 0:
        axs[i].set_ylabel('s=' + str(checkpoint['bottleneck_size']))

    if i >= 12:
        axs[i].set_xlabel('m=' + str(checkpoint['channel_multiplier']))

    axs[i].tick_params(labelbottom=False, labelleft=False, top=False, right=False, bottom=False, left=False)
    axs[i].imshow(output_image)
    # plot_image_tensor(output_tensor, ax=axs[i])

# plt.xlabel('Test')
# plt.ylabel('Test')
# plt.tight_layout()
plt.savefig('./images/output/samples/all.jpg', dpi=600)
plt.show()
