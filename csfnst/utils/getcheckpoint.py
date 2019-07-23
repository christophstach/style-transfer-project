import torch
from csfnst.fastneuralstyle.networks import TransformerNet, BottleneckType


def get_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = TransformerNet(
        channel_multiplier=checkpoint['channel_multiplier'],
        expansion_factor=checkpoint['expansion_factor'],
        bottleneck_type=BottleneckType[checkpoint['bottleneck_type'].replace('BottleneckType.', '')],
        bottleneck_size=checkpoint['bottleneck_size'],
        intermediate_activation_fn=checkpoint['intermediate_activation_fn'],
        final_activation_fn=checkpoint['final_activation_fn']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint, model
