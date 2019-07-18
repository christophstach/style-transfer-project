from os import listdir

import torch.cuda
import yaml

from csfnst.fastneuralstyle.networks import NetworkArchitecture, BottleneckType
from csfnst.utils import get_configs, get_model, get_criterion, get_dataloader, Trainer

run_configs_path = './config/'


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


for run_config_file in listdir(run_configs_path):
    run_configs_file = run_configs_path + run_config_file
    runs = yaml.load(open(run_configs_file, 'r'), Loader=yaml.Loader)
    configs = get_configs(runs)

    for config in configs:
        if config['active']:
            device = 'cuda'
            torch.cuda.empty_cache()

            model = get_model(config)
            dataloader = get_dataloader(config)
            criterion = get_criterion(config, device=device)

            trainer = Trainer(
                device=device,
                config=config,
                model=model,
                criterion=criterion,
                dataloader=dataloader,
                data_transformer=lambda batch: (batch[0], batch[0]),
                meta_data={
                    'attribution': config['attribution']
                }
            )

            trainer.load_checkpoint()
            trainer.train()

            del trainer
            del model
            del dataloader
            del criterion
