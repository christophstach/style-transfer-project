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

from csfnst.fastneuralstyle.networks import TransformerNet, BottleneckType
from csfnst.utils import save_image_tensor

CHECKPOINTS_PATH = './checkpoints'
UPLOAD_PATH = './temp/uploads'
GENERATED_IMAGE_PATH = './temp/generated'

rmtree(UPLOAD_PATH, ignore_errors=True)
rmtree(GENERATED_IMAGE_PATH, ignore_errors=True)

makedirs(UPLOAD_PATH)
makedirs(GENERATED_IMAGE_PATH)

models = dict()
styles = []


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

        models[style] = TransformerNet(
            channel_multiplier=checkpoint['channel_multiplier'],
            expansion_factor=checkpoint['expansion_factor'],
            bottleneck_type=BottleneckType[checkpoint['bottleneck_type'].replace('BottleneckType.', '')],
            bottleneck_size=checkpoint['bottleneck_size'],
            intermediate_activation_fn=checkpoint['intermediate_activation_fn'],
            final_activation_fn=checkpoint['final_activation_fn']
        )
        models[style].load_state_dict(checkpoint['model_state_dict'])
        styles.append({
            'id': style,
            'name': style,
            'contentImageSize': checkpoint['content_image_size'],
            'styleImageSize': checkpoint['style_image_size'],
            'contentWeight': checkpoint['content_weight'],
            'styleWeight': checkpoint['style_weight'],
            'totalVariationWeight': checkpoint['total_variation_weight'],
            'network': str(checkpoint['network']).replace('NetworkArchitecture.', ''),
            'bottleneckSize': checkpoint['bottleneck_size'],
            'bottleneckType': str(checkpoint['bottleneck_type']).replace('BottleneckType.', ''),
            'channelMultiplier': checkpoint['channel_multiplier'],
            'expansionFactor': checkpoint['expansion_factor'],
            'intermediateActivationFn': checkpoint['intermediate_activation_fn'],
            'finalActivationFn': checkpoint['final_activation_fn'],
            'metaData': {
                'attribution': {
                    'name': checkpoint.get('meta_data', {}).get('attribution', {}).get('name'),
                    'author': checkpoint.get('meta_data', {}).get('attribution', {}).get('author'),
                    'authorUrl': checkpoint.get('meta_data', {}).get('attribution', {}).get('author_url'),
                    'publishedUrl': checkpoint.get('meta_data', {}).get('attribution', {}).get('published_url'),
                    'publisher': checkpoint.get('meta_data', {}).get('attribution', {}).get('publisher'),
                    'publisherUrl': checkpoint.get('meta_data', {}).get('attribution', {}).get('publisher_url'),
                    'termsOfUseUrl': checkpoint.get('meta_data', {}).get('attribution', {}).get('terms_of_use_url')
                }
            }
        })

    return models[style]


def get_style_model(style):
    return models[style]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = GENERATED_IMAGE_PATH


@app.route('/style-transfer/generated/<path:image>', methods=['GET'])
def generated_images(image):
    # That is not secure
    return send_file('../' + GENERATED_IMAGE_PATH + '/' + image)


@app.route('/style-transfer/apply/<style>', methods=['POST'])
def apply(style):
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

    model = get_style_model(style).to(device)

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
    response = {
        'data': styles
    }

    return jsonify(response)


if __name__ == '__main__':
    device_type = 'cpu'

    model_names = listdir(CHECKPOINTS_PATH)
    model_names = [
        splitext(checkpoint_path)[0]
        for checkpoint_path in model_names
        if isfile(CHECKPOINTS_PATH + '/' + checkpoint_path)
           and '.pth' in checkpoint_path
           and 'experiment' not in checkpoint_path
    ]

    for path in model_names:
        load_style_model(path, device_type=device_type)

    app.run(host='0.0.0.0')
