import torchvision.transforms as transforms


def save_image_tensor(tensor, file_path):
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(file_path)
