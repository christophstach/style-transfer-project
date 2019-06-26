from csfnst.fastneuralstyle.networks import TransformerNet


def get_model(config):
    model = TransformerNet(
        channel_multiplier=config['channel_multiplier'],
        expansion_factor=config['expansion_factor'],
        bottleneck_type=config['bottleneck_type'],
        bottleneck_size=config['bottleneck_size'],
        intermediate_activation_fn=config['intermediate_activation_fn'],
        final_activation_fn=config['final_activation_fn']
    )

    return model
