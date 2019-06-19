import os.path
from os.path import splitext, basename
from time import time

import math
import logging
import torch
import torch.cuda
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from torch import autograd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from csfnst.fastneuralstyle.networks import TransformerNetConvTranspose, TransformerNetInterpolate, \
    TextureNet, NetworkArchitecture, TransformerNetConvTransposeLarge, StylizedNet, BottleneckType, \
    TransformerNetInterpolateLarge, StachMobileNet, StachMobileFullResidualNet, CustomTransformerNetInterpolate
from csfnst.losses import PerceptualLoss
from csfnst.utils import load_image, rename_network_layers, replace_network_layers

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(
        style_image_file,
        content_weight=1,
        style_weight=1e7,
        total_variation_weight=0,
        model_path=None,
        content_image_size=224,
        style_image_size=768,
        channel_multiplier=32,
        batch_size=18,
        epochs=10,
        save_checkpoint_interval=10,
        force_cpu=False,
        network_architecture=NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE,
        dataset_path='/home/ubuntu/.datasets/coco',
        learning_rate=1,
        augmentation=False,
        running_loss_range=50,
        normalize_gram=True,
        max_runtime=None,
        bottleneck_size=5,
        bottleneck_type=BottleneckType.RESIDUAL_BLOCK,
        expansion_factor=6,  # Mobile Version 2 only
        intermediate_activation_fn=None,
        final_activation_fn=None
):
    intermediate_activation_fn = intermediate_activation_fn if intermediate_activation_fn else 'ReLU6'
    final_activation_fn = final_activation_fn if final_activation_fn else 'Hardtanh'

    networks = {
        NetworkArchitecture.TEXTURE_NET: {
            'architecture': TextureNet(),
            'checkpoint_name': 'texture_net'
        },
        NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE: {
            'architecture': TransformerNetInterpolate(),
            'checkpoint_name': 'transformer_net_interpolate'
        },
        NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE: {
            'architecture': TransformerNetConvTranspose(),
            'checkpoint_name': 'transformer_net_conv_transpose'
        },
        NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE_LARGE: {
            'architecture': TransformerNetConvTransposeLarge(),
            'checkpoint_name': 'transformer_net_conv_transpose_large'
        },
        NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE_LARGE: {
            'architecture': TransformerNetInterpolateLarge(),
            'checkpoint_name': 'transformer_net_interpolate_large'
        },
        NetworkArchitecture.STACH_MOBILE_NET: {
            'architecture': StachMobileNet(),
            'checkpoint_name': 'stach_mobile_net'
        },
        NetworkArchitecture.STACH_MOBILE_FULL_RESIDUAL_NET: {
            'architecture': StachMobileFullResidualNet(),
            'checkpoint_name': 'stach_mobile_full_residual_net'
        },
        NetworkArchitecture.CUSTOM_TRANSFORMER_NET_INTERPOLATE: {
            'architecture': CustomTransformerNetInterpolate(channel_multiplier=channel_multiplier),
            'checkpoint_name': 'custom_transformer_net_interpolate'
        },
        NetworkArchitecture.STYLIZED_NET: {
            'architecture': StylizedNet(
                channel_multiplier=channel_multiplier,
                bottleneck_size=bottleneck_size,
                bottleneck_type=bottleneck_type,
                expansion_factor=expansion_factor,
                intermediate_activation_fn=intermediate_activation_fn,
                final_activation_fn=final_activation_fn
            ),
            'checkpoint_name': 'stylized_net'
        }
    }

    architecture = networks[network_architecture]['architecture']
    checkpoint_name = networks[network_architecture]['checkpoint_name']

    if not model_path:
        model_path = f'../checkpoints/{checkpoint_name}__'
        model_path += f'{splitext(basename(style_image_file))[0]}__'
        model_path += f'{content_image_size}__'
        model_path += f'{style_image_size}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')

    print(
        '\n#################################################################################\n\n' +
        'style_image_file=' + str(style_image_file) + '\n' +
        'content_weight=' + str(content_weight) + '\n' +
        'style_weight=' + str(style_weight) + '\n' +
        'total_variation_weight=' + str(total_variation_weight) + '\n' +
        'model_path=' + str(model_path) + '\n' +
        'content_image_size=' + str(content_image_size) + '\n' +
        'style_image_size=' + str(style_image_size) + '\n' +
        'batch_size=' + str(batch_size) + '\n' +
        'epochs=' + str(epochs) + '\n' +
        'save_checkpoint_interval=' + str(save_checkpoint_interval) + '\n' +
        'network_architecture=' + str(network_architecture) + '\n' +
        'force_cpu=' + str(force_cpu) + '\n' +
        'dataset_path=' + str(dataset_path) + '\n' +
        'learning_rate=' + str(learning_rate) + '\n' +
        'augmentation=' + str(augmentation) + '\n' +

        'channel_multiplier=' + str(channel_multiplier) + '\n' +
        'bottleneck_type=' + str(bottleneck_type) + '\n' +
        'bottleneck_size=' + str(bottleneck_size) + '\n' +
        'expansion_factor=' + str(expansion_factor) + '\n' +

        'intermediate_activation_fn=' + str(intermediate_activation_fn) + '\n' +
        'final_activation_fn=' + str(final_activation_fn) + '\n' +

        'running_loss_range=' + str(running_loss_range) + '\n' +
        'max_runtime=' + str(max_runtime) + '\n' +
        'Running on device: ' + str(device) + '\n\n' +
        '\n#################################################################################\n\n'
    )

    style_image = load_image(style_image_file, size=style_image_size).to(device)

    dataset = datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(content_image_size)
            if augmentation else
            transforms.Compose([
                transforms.Resize(content_image_size),
                transforms.CenterCrop(content_image_size)
            ]),
            transforms.ToTensor()
        ])
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    dataset_len = len(dataset)

    loss_network = rename_network_layers(
        replace_network_layers(
            models.vgg16(pretrained=True).features[:23],
            'MaxPool2d',
            'AvgPool2d'
        )
    ).to(device).eval()
    
    run_name = splitext(basename(model_path))[0]

    network = architecture.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=250, eta_min=0)
    writer = SummaryWriter(log_dir=f'/home/s0555912/runs/{run_name}')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        network.load_state_dict(checkpoint['network_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        content_loss_history = checkpoint['content_loss_history']
        style_loss_history = checkpoint['style_loss_history']
        total_variation_loss_history = checkpoint['total_variation_loss_history']
        loss_history = checkpoint['loss_history']
    else:
        content_loss_history = []
        style_loss_history = []
        total_variation_loss_history = []
        loss_history = []

    network.train()
    criterion = PerceptualLoss(
        model=loss_network,
        content_layers=['relu3_3'],
        style_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
        style_image=style_image,
        content_weight=content_weight,
        style_weight=style_weight,
        total_variation_weight=total_variation_weight,
        normalize_gram=normalize_gram,
        content_loss_history=content_loss_history,
        style_loss_history=style_loss_history,
        total_variation_loss_history=total_variation_loss_history,
        loss_history=loss_history
    )

    start = time()
    for epoch in range(epochs):

        progress_bar = tqdm(
            enumerate(dataloader, 0),
            total=math.ceil(dataset_len / batch_size)
        )

        epoch_loss_history = []

        for i, batch in progress_bar:
            with autograd.detect_anomaly():
                try:
                    images, _ = batch
                    images = images.to(device)
                    optimizer.zero_grad()

                    predictions = network(images)

                    loss = criterion(predictions, images)
                    loss.backward()
                    optimizer.step()

                    epoch_loss_history.append(loss.item())
                    
                    if True: # Use tensorboard
                        grid_inputs = torchvision.utils.make_grid(images)
                        grid_predictions = torchvision.utils.make_grid(predictions)

                        writer.add_image('Inputs', grid_inputs, 0)
                        writer.add_image('Predictions', grid_predictions, 0)

                        # writer.add_graph(network, images)
                        writer.add_scalar(
                            'Content Loss', 
                            criterion.content_loss_history[-1], 
                            len(criterion.content_loss_history) - 1
                        )
                        
                        writer.add_scalar(
                            'Style Loss', 
                            criterion.style_loss_history[-1], 
                            len(criterion.style_loss_history) - 1 
                        )
                        
                        writer.add_scalar(
                            'TV Loss', 
                            criterion.total_variation_loss_history[-1], 
                            len(criterion.total_variation_loss_history) - 1
                        )
                        
                        writer.add_scalar(
                            'Total Loss', 
                            criterion.loss_history[-1], 
                            len(criterion.loss_history) - 1
                        )
                        
                        writer.close()
                    


                    if i % save_checkpoint_interval == 0:
                        torch.save({
                            'network_state_dict': network.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'content_loss_history': criterion.content_loss_history,
                            'style_loss_history': criterion.style_loss_history,
                            'total_variation_loss_history': criterion.total_variation_loss_history,
                            'loss_history': criterion.loss_history,
                            'content_image_size': content_image_size,
                            'style_image_size': style_image_size,
                            'network_architecture': str(network_architecture),
                            'content_weight': content_weight,
                            'style_weight': style_weight,
                            'total_variation_weight': total_variation_weight,
                            'bottleneck_size': bottleneck_size,
                            'bottleneck_type': str(bottleneck_type),
                            'channel_multiplier': channel_multiplier,
                            'expansion_factor': expansion_factor,
                            'intermediate_activation_fn': intermediate_activation_fn,
                            'final_activation_fn': final_activation_fn
                        }, model_path)
                except Exception as e:
                    print(e)
                    pass

            avg_epoch_loss = sum(epoch_loss_history) / (i + 1)

            if len(criterion.loss_history) >= running_loss_range:
                running_loss = sum(criterion.loss_history[-running_loss_range:]) / running_loss_range
            else:
                running_loss = 0

            if len(criterion.loss_history) > 0:
                progress_bar.set_description(
                    f'Epoch: {epoch + 1}/{epochs}, ' +
                    f'Average Epoch Loss: {avg_epoch_loss:,.2f}, ' +
                    f'Running Loss: {running_loss:,.2f}, ' +
                    f'Loss: {criterion.loss_history[-1]:,.2f}, ' +
                    f'Learning Rate: {scheduler.get_lr()[0]:,.6f}'
                )
                scheduler.step()

            if time() - start >= max_runtime:
                break

        if time() - start >= max_runtime:
            break
