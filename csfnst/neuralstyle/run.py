import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

from csfnst.layers import Normalization
from csfnst.losses import PerceptualLoss
from csfnst.utils import load_image, save_image_tensor, plot_image_tensor


def run(
        content_image_file,
        style_image_file,
        output_image_file,
        image_size=(320, 320),
        epochs=25,
        content_weight=1,
        style_weight=1e-5,
        force_cpu=False
):
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    style_image = load_image(style_image_file, size=image_size).to(device)
    content_image = load_image(content_image_file, size=image_size).to(device)
    output_image = content_image.clone().to(device)

    normalization_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    model = nn.Sequential(
        Normalization(normalization_mean, normalization_std),
        *models.vgg16(pretrained=True).features.to(device).eval()[:23]
    ).to(device)

    output_image.requires_grad_()

    criterion = PerceptualLoss(
        model=model,
        content_layers=[16],
        style_layers=[4, 9, 16, 23],
        style_image=style_image,
        content_weight=content_weight,
        style_weight=style_weight
    )

    optimizer = optim.LBFGS([output_image])
    progress_bar = tqdm(range(epochs))

    for _ in progress_bar:
        def closure():
            optimizer.zero_grad()
            loss = criterion(output_image.unsqueeze(0), content_image.unsqueeze(0))
            loss.backward()

            progress_bar.set_description(f'Loss: {round(loss.item(), 5):,}')
            return loss

        optimizer.step(closure)

    output_image.data.clamp_(0, 1)
    save_image_tensor(output_image, output_image_file)

    print(f'Saved file to {output_image_file}')
    plot_image_tensor(output_image, show=True)
