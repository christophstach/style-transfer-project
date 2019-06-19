from os import listdir

import yaml

from csfnst.fastneuralstyle import train_fast_neural_style
from csfnst.fastneuralstyle.networks import NetworkArchitecture, BottleneckType

configs_path = '../config/'


def get_runs(config_file):
    config = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)

    runs = [
        {
            'name': list(run.keys())[0],
            **config['defaults'],
            **run[list(run.keys())[0]],
            'network': NetworkArchitecture[
                run[list(run.keys())[0]]['network']
                if 'network' in run[list(run.keys())[0]].keys()
                else config['defaults']['network']
            ],
            'bottleneck_type': BottleneckType[
                run[list(run.keys())[0]]['bottleneck_type']
                if 'bottleneck_type' in run[list(run.keys())[0]].keys()
                else config['defaults']['bottleneck_type']
            ]
        }
        for run in config['runs']
    ]

    return runs


for config_file in listdir(configs_path):
    config_path = configs_path + config_file

    for r in get_runs(config_path):
        if r['active']:
            train_fast_neural_style(
                style_image_file=f'../images/style/{r["style_image"]}',
                dataset_path=f'{r["dataset_path"]}{r["dataset"]}',
                model_path=f'../checkpoints/{r["name"]}.pth',
                content_weight=float(r['content_weight']),
                style_weight=float(r['style_weight']),
                total_variation_weight=float(r['total_variation_weight']),
                epochs=int(r['epochs']),
                batch_size=int(r['batch_size']),
                content_image_size=int(r['content_image_size']),
                style_image_size=int(r['style_image_size']),
                learning_rate=float(r['learning_rate']),
                network_architecture=r['network'],
                bottleneck_type=r['bottleneck_type'],
                bottleneck_size=int(r['bottleneck_size']),
                save_checkpoint_interval=int(r['save_checkpoint_interval']),
                expansion_factor=int(r['expansion_factor']),
                augmentation=bool(r['augmentation']),
                channel_multiplier=int(r['channel_multiplier']),
                max_runtime=float(r['max_runtime']),
                intermediate_activation_fn=str(r['intermediate_activation_fn']),
                final_activation_fn=str(r['final_activation_fn'])
            )
