import os

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from csfnst.utils import load_image, get_checkpoint, flatten

experiment = 'experiment2'
image = load_image('./images/content/htw-768x768.jpg').unsqueeze(0)
output_image_path = './images/experiments/'
checkpoints_path = './checkpoints/'
checkpoints = [
    checkpoint
    for checkpoint in os.listdir(checkpoints_path)
    if experiment in checkpoint
]
checkpoints.sort()

fig, axs = plt.subplots(4, 4, figsize=(8, 8), sharex='row', sharey='col')
fig.subplots_adjust(wspace=.025, hspace=.025)
axs = flatten(axs)

for i, checkpoint in enumerate(checkpoints):
    checkpoint, model = get_checkpoint(f'{checkpoints_path}{checkpoint}')
    output_tensor = model(image).squeeze()
    output_image = transforms.ToPILImage()(output_tensor.cpu())

    if i % 4 == 0:
        axs[i].set_ylabel('s=' + str(checkpoint['bottleneck_size']))

    if i >= 12:
        axs[i].set_xlabel('m=' + str(checkpoint['channel_multiplier']))

    axs[i].tick_params(labelbottom=False, labelleft=False, top=False, right=False, bottom=False, left=False)
    axs[i].imshow(output_image)

plt.savefig(f'{output_image_path}fast_image_grid_{experiment}.jpg', dpi=600)
plt.show()
