import torchvision.transforms as transforms
from PIL import Image


def load_image(file_path, size=None, normalize=False):
    if normalize:
        if size:
            transform = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        if size:
            transform = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

    image = Image.open(file_path)

    return transform(image)
