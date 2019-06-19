import yaml

from csfnst.utils import get_config, get_model, get_criterion, get_dataloader, Trainer

runs_file = '../config/production.yml'
runs = yaml.load(open(runs_file, 'r'), Loader=yaml.Loader)
config = get_config(runs, 'candy')

config['batch_size'] = 1
config['content_image_size'] = 64
config['name'] = 'debugger'

device = 'cuda'

model = get_model(config)
dataloader = get_dataloader(config)
criterion = get_criterion(config, device=device)

trainer = Trainer(
    device=device,
    config=config,
    model=model,
    criterion=criterion,
    dataloader=dataloader,
    dataloader_transformer=lambda dl: ((images, images) for images, _ in dl)
)

trainer.train()
