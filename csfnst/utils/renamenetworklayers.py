from collections import OrderedDict

from torch.nn import Sequential


def rename_network_layers(network):
    network_dict = []

    mapper = {
        'Conv2d': 'conv',
        'MaxPool2d': 'maxp',
        'AvgPool2d': 'avgp'
    }

    pooling_types = ['MaxPool2d', 'AvgPool2d']

    block_number = 1
    layer_numbers = {}

    for i, layer in enumerate(network):
        if type(layer).__name__ in mapper:
            layer_type = mapper[type(layer).__name__]
        else:
            layer_type = str(type(layer).__name__).lower()

        if layer_type in layer_numbers.keys():
            layer_numbers[layer_type] += 1
        else:
            layer_numbers[layer_type] = 1

        if type(layer).__name__ in pooling_types:
            layer_name = f'{layer_type}{block_number}'
        else:
            layer_name = f'{layer_type}{block_number}_{layer_numbers[layer_type]}'

        network_dict.append((
            layer_name,
            layer
        ))

        if type(layer).__name__ in pooling_types:
            block_number += 1

            for i in layer_numbers.keys():
                layer_numbers[i] = 0

    return Sequential(OrderedDict(network_dict))
