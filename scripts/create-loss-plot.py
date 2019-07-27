import os

import matplotlib.pyplot as plt
import numpy as np

from csfnst.utils import get_checkpoint


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


running_loss_range = 100
experiment = 'experiment2'
output_image_path = './images/experiments/'
checkpoints_path = './checkpoints/'
checkpoints = [
    checkpoint
    for checkpoint in os.listdir(checkpoints_path)
    if experiment in checkpoint
]
checkpoints.sort()

fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex='row', sharey='col')
fig.subplots_adjust(wspace=.025, hspace=.025)
legend_items = []
for i, checkpoint in enumerate(checkpoints):
    checkpoint, _ = get_checkpoint(f'{checkpoints_path}{checkpoint}')
    s = checkpoint['bottleneck_size']
    m = checkpoint['channel_multiplier']

    running_losses = running_mean(np.array(checkpoint['loss_history']), running_loss_range)

    if int(m) == 16:
        ax.semilogy(running_losses, linewidth=1.5)
    else:
        ax.semilogy(running_losses, linewidth=0.5)

    legend_items.append(f'Network {i + 1} (s={s}, m={m})')

plt.legend(legend_items)
plt.savefig(f'{output_image_path}fast_loss_plot_{experiment}.jpg', dpi=400)
plt.show()
