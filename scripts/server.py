from os import listdir, makedirs
from os.path import isfile, splitext, basename
from shutil import rmtree
from time import time
from uuid import uuid4

import torch
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from csfnst.fastneuralstyle.networks import StylizedNet, BottleneckType
from csfnst.utils import save_image_tensor

CHECKPOINTS_PATH = './checkpoints'
UPLOAD_PATH = './temp/uploads'
GENERATED_IMAGE_PATH = './temp/generated'

rmtree(UPLOAD_PATH, ignore_errors=True)
rmtree(GENERATED_IMAGE_PATH, ignore_errors=True)

makedirs(UPLOAD_PATH)
makedirs(GENERATED_IMAGE_PATH)

models = dict()
checkpoint_paths = listdir(CHECKPOINTS_PATH)
checkpoint_paths = [
    checkpoint_path
    for checkpoint_path in checkpoint_paths
    if isfile(CHECKPOINTS_PATH + checkpoint_path)
]


def checkpoint2uri(checkpoint_path):
    name, ext = splitext(checkpoint_path)

    return name.replace('_', '-')


def uri2checkpoint(uri):
    name = uri.replace('-', '_')
    return f'{CHECKPOINTS_PATH}/{name}.pth'


def load_style_model(style, device_type):
    if style not in models.keys():
        if device_type == 'cuda':
            checkpoint = torch.load(uri2checkpoint(style))
        else:
            checkpoint = torch.load(uri2checkpoint(style), map_location={'cuda:0': 'cpu'})

        models[style] = StylizedNet(
            channel_multiplier=checkpoint['channel_multiplier'],
            expansion_factor=checkpoint['expansion_factor'],
            bottleneck_type=BottleneckType[checkpoint['bottleneck_type'].replace('BottleneckType.', '')],
            bottleneck_size=checkpoint['bottleneck_size'],
            intermediate_activation_fn=checkpoint['intermediate_activation_fn'],
            final_activation_fn=checkpoint['final_activation_fn']
        )
        models[style].load_state_dict(checkpoint['model_state_dict'])

    return models[style]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = GENERATED_IMAGE_PATH
CORS(app)


@app.route('/style-transfer/generated/<path:image>', methods=['GET'])
def generated_images(image):
    # That is not secure
    return send_file('../' + GENERATED_IMAGE_PATH + '/' + image)


@app.route('/style-transfer/apply/<device_type>/<style>', methods=['POST'])
def apply(device_type, style):
    device = torch.device(device_type)

    input_image = request.files['image']
    input_image_path = UPLOAD_PATH + '/' + input_image.filename
    input_image.save(input_image_path)

    image = Image.open(input_image_path)
    width, height = image.size
    max = 1440

    if width >= height:
        if width > max:
            image = image.resize((max, int(max * (height / width))))
    else:
        if height >= max:
            image = image.resize((int(max * (width / height)), max))

    input_tensor = transforms.ToTensor()(image).to(device).unsqueeze(0)

    model = load_style_model(style, device_type).to(device)

    start = time()
    output_tensor = model(input_tensor)
    end = time()

    output_tensor = output_tensor.squeeze()

    generated_image_path = f'{GENERATED_IMAGE_PATH}/{style}--{uuid4()}.jpg'
    generated_image_uri = basename(generated_image_path)
    save_image_tensor(output_tensor, generated_image_path)

    response = {
        'data': {
            'timeElapsed': (end - start),
            'styledImageUrl': f'/style-transfer/generated/{generated_image_uri}'
        }
    }

    return jsonify(response)


@app.route('/style-transfer/list-styles', methods=['GET'])
def list_styles():
    checkpoints = listdir(CHECKPOINTS_PATH)
    checkpoints = [
        {
            'id': checkpoint2uri(checkpoint),
            'name': checkpoint2uri(checkpoint)
        }
        for checkpoint in checkpoints
        if isfile(f'{CHECKPOINTS_PATH}/{checkpoint}')
    ]

    response = {
        'data': checkpoints
    }

    return jsonify(response)


def list_styles2():
    checkpoints = listdir(CHECKPOINTS_PATH)
    checkpoints = [
        {
            'name': checkpoint2uri(checkpoint),
            'data': torch.load(f'{CHECKPOINTS_PATH}/{checkpoint}', map_location={'cuda:0': 'cpu'})
        }
        for checkpoint in checkpoints
        if isfile(f'{CHECKPOINTS_PATH}/{checkpoint}') and '__' not in checkpoint
    ]

    checkpoints = [
        {
            'id': checkpoint['name'],
            'name': checkpoint['name'],
            'contentImageSize': checkpoint['data']['content_image_size'],
            'styleImageSize': checkpoint['data']['style_image_size'],
            'contentWeight': checkpoint['data']['content_weight'],
            'styleWeight': checkpoint['data']['style_weight'],
            'totalVariationWeight': checkpoint['data']['total_variation_weight'],
            'network': checkpoint['data']['network'],
            'bottleneckSize': checkpoint['data']['bottleneck_size'],
            'bottleneckType': checkpoint['data']['bottleneck_type'],
            'channelMultiplier': checkpoint['data']['channel_multiplier'],
            'expansionFactor': checkpoint['data']['expansion_factor'],
            'intermediateActivationFn': checkpoint['data']['intermediate_activation_fn'],
            'finalActivationFn': checkpoint['data']['final_activation_fn']

        }
        for checkpoint in checkpoints
    ]

    response = {
        'data': checkpoints
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
