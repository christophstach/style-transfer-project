from collections import OrderedDict
import torch.nn as nn


def replace_network_layers(network, layer_old, layer_new):
    supported_layers = ['MaxPool2d', 'AvgPool2d']

    layers_map = {
        'MaxPool2d': nn.MaxPool2d,
        'AvgPool2d': nn.AvgPool2d
    }

    assert layer_old in supported_layers, 'Layer type not supported'
    assert layer_new in supported_layers, 'Layer type not supported'

    network_dict = []

    for i, layer in enumerate(network):
        if type(layer).__name__ == layer_old:
            assert not (type(layer).__name__ == 'MaxPool2d' and layer.dilation != 1), 'Dilation is not supported'

            layer = layers_map[layer_new](
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                ceil_mode=layer.ceil_mode
            )

        network_dict.append((str(i), layer))

    return nn.Sequential(OrderedDict(network_dict))
