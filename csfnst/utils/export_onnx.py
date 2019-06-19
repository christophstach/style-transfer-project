from os.path import basename, splitext, exists

import torch
import torch.cuda
import torch.onnx

from csfnst.fastneuralstyle.networks import NetworkArchitecture, TransformerNetConvTranspose, TransformerNetInterpolate, \
    TextureNet, TransformerNetInterpolateLarge, TransformerNetConvTransposeLarge, \
    StachMobileFullResidualNet, StachMobileNet, CustomTransformerNetInterpolate


def export_onnx(
        network,
        input_model_path=None,
        output_model_path=None,
        image_size=None,
        force_cpu=True
):
    assert exists(input_model_path), 'Model weights not found!'

    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    model_name, _ = splitext(basename(input_model_path))

    if not image_size and model_name.count('__') == 2:
        name, image_sizes, hyper_params = model_name.split('__')
        content_image_size, style_image_size = image_sizes.split('_')

        content_image_size = int(content_image_size.replace('c', ''))
        style_image_size = int(style_image_size.replace('s', ''))
    else:
        content_image_size = 1024

    if not output_model_path:
        output_model_path = f'../onnx/{model_name}.onnx'

    dummy_input = torch.randn((1, 3, content_image_size, content_image_size)).to(device)
    transformer_network = network.to(device)

    if force_cpu:
        checkpoint = torch.load(input_model_path, map_location={'cuda:0': 'cpu'})
    else:
        checkpoint = torch.load(input_model_path)

    transformer_network.load_state_dict(checkpoint['model_state_dict'])
    transformer_network.eval()

    torch.onnx.export(transformer_network, dummy_input, output_model_path, verbose=True)
