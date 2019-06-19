import os
from time import time

import torch
import torch.cuda

from csfnst.fastneuralstyle.networks import NetworkArchitecture, TransformerNetConvTranspose, \
    TransformerNetInterpolate, TextureNet
from csfnst.utils import load_image, plot_image_tensor, save_image_tensor


def run(
        model_path,
        image_size,
        content_image_file,
        output_image_file,
        network_architecture=NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE,
        force_cpu=True,
        save_image=False
):
    assert os.path.exists(model_path), 'Model weights not found!'

    networks = {
        NetworkArchitecture.TEXTURE_NET: TextureNet,
        NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE: TransformerNetInterpolate,
        NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE: TransformerNetConvTranspose
    }

    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    content_image = load_image(content_image_file, size=(image_size, image_size)).to(device)
    network = networks[network_architecture]().to(device)

    if force_cpu:
        checkpoint = torch.load(model_path, map_location={'cuda:0': 'cpu'})
    else:
        checkpoint = torch.load(model_path)

    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    start = time()
    output_image = network(content_image.unsqueeze(0)).squeeze()
    end = time()

    print(f'Time elapsed calculating the new image: {round(end - start, 2)} sec')

    if save_image:
        save_image_tensor(output_image, output_image_file)
        print(f'Saved file to {output_image_file}')

    print(output_image.shape)
    plot_image_tensor(output_image, show=True)
