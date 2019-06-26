import torch

from csfnst.fastneuralstyle.networks import TransformerNet
from torchsummary import summary

network1 = TransformerNet(bottleneck_size=5, channel_multiplier=32)
network2 = TransformerNet(bottleneck_size=5, channel_multiplier=16)
network3 = TransformerNet(bottleneck_size=5, channel_multiplier=8)

network4 = TransformerNet(bottleneck_size=4, channel_multiplier=32)
network5 = TransformerNet(bottleneck_size=4, channel_multiplier=16)
network6 = TransformerNet(bottleneck_size=4, channel_multiplier=8)

network7 = TransformerNet(bottleneck_size=3, channel_multiplier=32)
network8 = TransformerNet(bottleneck_size=3, channel_multiplier=16)
network9 = TransformerNet(bottleneck_size=3, channel_multiplier=8)

device_type = 'cpu'
device = torch.device(device_type)
model = network9.to(device)

summary(model, (3, 1024, 768), device=device_type)
