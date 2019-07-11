import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataloader(config):
    dataset = datasets.ImageFolder(
        config['dataset_path'] + config['dataset'],
        transform=transforms.Compose([
            transforms.Compose([
                transforms.Resize(255),
                transforms.RandomResizedCrop(config['content_image_size']),
                transforms.RandomRotation([0, 10])
            ])
            if config['augmentation'] else
            transforms.Compose([
                transforms.Resize(config['content_image_size']),
                transforms.CenterCrop(config['content_image_size'])
            ]),
            transforms.ToTensor()
        ]),

    )

    dataloader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    return dataloader
