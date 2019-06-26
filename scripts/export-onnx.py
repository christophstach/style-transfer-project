import os.path

import torch

from csfnst.fastneuralstyle.networks import TransformerNet, BottleneckType
from csfnst.utils import export_onnx


checkpoints_path = '../checkpoints/'

models = os.listdir(checkpoints_path)
models = [model for model in models if os.path.isfile(checkpoints_path + model)]

while True:
    for i, model in enumerate(models):
        print(f'{i + 1}: {model}')

    print('\n')

    try:
        decision = int(input('Choose your model: ')) - 1
        print(f'Your decision was {models[decision]}')

        break
    except ValueError:
        pass
    except IndexError:
        pass

checkpoint = torch.load(checkpoints_path + models[decision])

model = TransformerNet(
    channel_multiplier=checkpoint['channel_multiplier'],
    bottleneck_size=checkpoint['bottleneck_size'],
    bottleneck_type=BottleneckType[checkpoint['bottleneck_type'].replace('BottleneckType.', '')],
    expansion_factor=checkpoint['expansion_factor'],
    intermediate_activation_fn=checkpoint['intermediate_activation_fn'],
    final_activation_fn=checkpoint['final_activation_fn']
)

onnx_model_path = export_onnx(
    input_model_path=checkpoints_path + models[decision],
    network=model
)
