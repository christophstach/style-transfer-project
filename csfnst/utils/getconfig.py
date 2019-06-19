from csfnst.fastneuralstyle.networks import NetworkArchitecture, BottleneckType


def get_configs(runs):
    runs = [
        {
            'name': list(run.keys())[0],
            **runs['defaults'],
            **run[list(run.keys())[0]],
            'network': NetworkArchitecture[
                run[list(run.keys())[0]]['network']
                if 'network' in run[list(run.keys())[0]].keys()
                else runs['defaults']['network']
            ],
            'bottleneck_type': BottleneckType[
                run[list(run.keys())[0]]['bottleneck_type']
                if 'bottleneck_type' in run[list(run.keys())[0]].keys()
                else runs['defaults']['bottleneck_type']
            ]
        }
        for run in runs['runs']
    ]

    return [
        {
            **run,
            'epochs': int(run['epochs']),
            'min_lr': float(run['min_lr']),
            'max_lr': float(run['max_lr']),
            'lr_multiplicator': run['lr_multiplicator'],
            'content_weight': float(run['content_weight']),
            'style_weight': float(run['style_weight']),
            'total_variation_weight': float(run['total_variation_weight'])
        } for run in runs
    ]


def get_config(run, name):
    configs = get_configs(run)

    return [config for config in configs if config['name'] == name][0]
