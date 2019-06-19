import os.path

import math
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from csfnst.fastneuralstyle.networks import NetworkArchitecture, TransformerNetInterpolate
from csfnst.losses import PerceptualLoss
from csfnst.utils import load_image, rename_network_layers, replace_network_layers, plot_image_tensor

style_image_file = '../images/style/starry_night_google.jpg'
content_weight = 1
style_weight = 1e7
content_image_size = 224
style_image_size = 768
batch_size = 1
epochs = 1000
save_checkpoint_interval = 10
force_cpu = False
network_architecture = NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE_LARGE
dataset_path = '/home/christoph/.datasets/coco_one_file'
learning_rate = 10
augmentation = False

architecture = TransformerNetInterpolate
model_path = f'../checkpoints/debug.pth'
device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')

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
    num_workers=2
)

dataset_len = len(dataset)

loss_network = rename_network_layers(
    replace_network_layers(
        models.vgg16(pretrained=True).features[:23],
        'MaxPool2d',
        'AvgPool2d'
    )
).to(device).eval()

network = architecture().to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
loss_history = []

if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['network_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history = checkpoint['loss_history']

network.train()
criterion = PerceptualLoss(
    model=loss_network,
    content_layers=['relu3_3'],
    style_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
    style_image=style_image,
    content_weight=content_weight,
    style_weight=style_weight,
    normalize_gram=True
)

# plot_image_tensor(style_image, show=True)

for epoch in range(epochs):
    progress_bar = tqdm(
        enumerate(dataloader, 0),
        total=math.ceil(dataset_len / batch_size)
    )

    epoch_loss_history = []

    for i, batch in progress_bar:
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()

        predictions = network(images)

        loss = criterion(predictions, images)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        epoch_loss_history.append(loss.item())

        if i % save_checkpoint_interval == 0:
            torch.save({
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history
            }, model_path)

        avg_epoch_loss = sum(epoch_loss_history) / (i + 1)
        progress_bar.set_description(
            f'Epoch: {epoch + 1}/{epochs}, Average Epoch Loss: {avg_epoch_loss:,.2f}, Loss: {loss_history[-1]:,.2f}'
        )

network.eval()

images, _ = next(iter(dataloader))
images = images.to(device)

# print(images.shape)
# plot_image_tensor(images.squeeze(), show=True)

predictions = network(images)

# print(predictions.shape)
plot_image_tensor(predictions.squeeze(), show=True)
