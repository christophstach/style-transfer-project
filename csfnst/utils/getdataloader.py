import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataloader(config):
    dataset = datasets.ImageFolder(
        config['dataset_path'] + config['dataset'],
        transform=transforms.Compose([
            transforms.RandomResizedCrop(config['content_image_size'])
            if config['augmentation'] else
            transforms.Compose([
                transforms.Resize(config['content_image_size']),
                transforms.CenterCrop(config['content_image_size'])
            ]),
            transforms.ToTensor()
            # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),

    )

    dataloader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    return dataloader
