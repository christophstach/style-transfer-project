from csfnst.fastneuralstyle import train_fast_neural_style
from csfnst.fastneuralstyle.networks import NetworkArchitecture

"""
train_fast_neural_style(
    style_image_file='../images/style/starry_night_google.jpg',
    dataset_path='/home/christoph/.datasets/coco_tiny',
    epochs=50,
    batch_size=5,
    content_image_size=128,
    style_image_size=128,
    learning_rate=1e-6,
    network_architecture=NetworkArchitecture.TRANSFORMER_NET_CONV_TRANSPOSE
)
"""

train_fast_neural_style(
    style_image_file='../images/style/starry_night_google.jpg',
    dataset_path='/home/christoph/.datasets/coco',
    epochs=1,
    batch_size=2,
    content_image_size=224,
    style_image_size=768,
    learning_rate=1e-3,
    network_architecture=NetworkArchitecture.TRANSFORMER_NET_INTERPOLATE_LARGE,
    augmentation=False
)
