from csfnst.fastneuralstyle import train_fast_neural_style
from csfnst.fastneuralstyle.networks import NetworkArchitecture


train_fast_neural_style(
    style_image_file='../images/style/starry_night_google.jpg',  # Doesn't matter
    model_path='../checkpoints/pretrained/transformer_net_interpolate_large_pretrained_360.pth',
    dataset_path='/home/christoph/.datasets/coco',
    epochs=500,
    batch_size=1,
    content_image_size=360,
    style_image_size=1,
    learning_rate=1e-3,
    network_architecture=NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE_LARGE,
    augmentation=False,
    style_weight=0,
    content_weight=1
)
