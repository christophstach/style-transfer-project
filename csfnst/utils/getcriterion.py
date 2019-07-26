import torch
import torchvision.models as models

from csfnst.losses import PerceptualLoss
from csfnst.utils import load_image, rename_network_layers, replace_network_layers


def get_last_layer(layers, vgg19=False):
    layers.sort()
    last_layer = layers[-1:][0]
    layer_type = last_layer[:4]

    block, unit = last_layer.replace(layer_type, '').split('_')
    block = int(block)
    unit = int(unit)

    if block <= 2:
        last_layer_number = (block - 1) * 5 + unit * 2
    else:
        if vgg19:
            last_layer_number = 10 + (block - 3) * 9 + unit * 2
        else:
            last_layer_number = 10 + (block - 3) * 7 + unit * 2

    last_layer_number = last_layer_number - 1 if layer_type == 'conv' else last_layer_number

    return last_layer_number


def get_criterion(config, device):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        style_image_file = f'./images/style/{config["style_image"]}'
        style_image = load_image(style_image_file, size=config['style_image_size']).to(device)
        last_layer = get_last_layer(list(set(config['content_layers'] + config['style_layers'])),
                                    config['loss_network'] == 'vgg19')

        loss_network = rename_network_layers(
            replace_network_layers(
                (
                    models.vgg16(pretrained=True).features[:last_layer]
                    if config['loss_network'] == 'vgg16'
                    else models.vgg19(pretrained=True).features[:last_layer]
                ),
                'MaxPool2d',
                'AvgPool2d'
            )
        ).to(device).eval()

        criterion = PerceptualLoss(
            model=loss_network,
            content_layers=config['content_layers'],
            style_layers=config['style_layers'],
            style_image=style_image,
            content_weight=config['content_weight'],
            style_weight=config['style_weight'],
            total_variation_weight=config['total_variation_weight']
        )

        return criterion
