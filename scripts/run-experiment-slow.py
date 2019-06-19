from os.path import basename, splitext

import torch
import torch.optim as optim
from tqdm import tqdm

from csfnst.utils import get_criterion
from csfnst.utils import load_image, save_image_tensor


def run_experiment(
        style_image_file,
        content_image_size,
        style_weight,
        tv_weight,
        epochs=5,
        prefix='htw',
        content_image_file='../images/content/htw-768x768.jpg',
        use_random_noise=False,
        use_lbfgs=True
):
    output_image_file = f'../images/experiments/'
    output_image_file += f'{prefix}__{splitext(basename(style_image_file))[0]}'
    output_image_file += f'__{content_image_size}x{content_image_size}'
    output_image_file += f'__style-weight_{style_weight:,.0e}__tv-weight_{tv_weight:,.0e}.jpg'

    force_cpu = False
    style_image_size = 768

    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    content_image = load_image(content_image_file, size=content_image_size, normalize=False).to(device)

    if use_random_noise:
        output_image = torch.rand(content_image.shape[0], content_image.shape[1], content_image.shape[2]).to(device)
    else:
        output_image = content_image.clone().to(device)

    config = {
        'loss_network': 'vgg16',
        'content_weight': 1,
        'style_weight': style_weight,
        'total_variation_weight': tv_weight,
        'style_image': style_image_file,
        'style_image_size': style_image_size,
        'content_layers': ['relu3_3'],
        'style_layers': ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    }

    criterion = get_criterion(config, device='cpu' if force_cpu else 'cuda')
    optimizer = optim.LBFGS([output_image]) if use_lbfgs else optim.Adam([output_image], lr=1e-1)
    content_image.unsqueeze_(0)

    output_image.unsqueeze_(0)
    output_image.requires_grad_()

    content_loss_history = []
    style_loss_history = []
    total_variation_loss_history = []
    loss_history = []

    progress_bar = tqdm(range(epochs))

    if use_lbfgs:
        for epoch in progress_bar:
            def closure():
                output_image.data.clamp_(0, 1)
                optimizer.zero_grad()

                loss = criterion(output_image, content_image)
                loss.backward()

                content_loss_history.append(criterion.content_loss_val)
                style_loss_history.append(criterion.style_loss_val)
                total_variation_loss_history.append(criterion.total_variation_loss_val)
                loss_history.append(criterion.loss_val)

                progress_bar.set_description(f'Loss: {loss.item():,.2f}')

                return loss

            optimizer.step(closure)
    else:
        for epoch in progress_bar:
            output_image.data.clamp_(0, 1)
            optimizer.zero_grad()

            loss = criterion(output_image, content_image)
            loss.backward()

            content_loss_history.append(criterion.content_loss_val)
            style_loss_history.append(criterion.style_loss_val)
            total_variation_loss_history.append(criterion.total_variation_loss_val)
            loss_history.append(criterion.loss_val)

            progress_bar.set_description(f'Loss: {loss.item():,.2f}')

            optimizer.step()

    content_image.squeeze_()

    output_image.detach_()
    output_image.squeeze_()
    output_image.data.clamp_(0, 1)
    save_image_tensor(output_image, output_image_file)


# Style
run_experiment(prefix='a', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e5, tv_weight=0)
run_experiment(prefix='a', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e6, tv_weight=0)
run_experiment(prefix='a', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=0)
run_experiment(prefix='a', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e8, tv_weight=0)
run_experiment(prefix='a', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e9, tv_weight=0)

run_experiment(prefix='a', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e5, tv_weight=0)
run_experiment(prefix='a', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e6, tv_weight=0)
run_experiment(prefix='a', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e7, tv_weight=0)
run_experiment(prefix='a', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=0)
run_experiment(prefix='a', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e9, tv_weight=0)

# Total-Variation
run_experiment(prefix='b', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=1e-7)
run_experiment(prefix='b', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=1e-6)
run_experiment(prefix='b', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=1e-5)
run_experiment(prefix='b', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=1e-4)
run_experiment(prefix='b', style_image_file='the_scream.jpg', content_image_size=768, style_weight=1e7, tv_weight=1e-3)

run_experiment(prefix='b', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=1e-7)
run_experiment(prefix='b', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=1e-6)
run_experiment(prefix='b', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=1e-5)
run_experiment(prefix='b', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=1e-4)
run_experiment(prefix='b', style_image_file='starry_night.jpg', content_image_size=768, style_weight=1e8, tv_weight=1e-3)
