import torch

from csfnst.fastneuralstyle.networks import StylizedNet
from torchsummary import summary

network1 = StylizedNet(bottleneck_size=5, channel_multiplier=32)
network2 = StylizedNet(bottleneck_size=5, channel_multiplier=16)
network3 = StylizedNet(bottleneck_size=5, channel_multiplier=8)

network4 = StylizedNet(bottleneck_size=4, channel_multiplier=32)
network5 = StylizedNet(bottleneck_size=4, channel_multiplier=16)
network6 = StylizedNet(bottleneck_size=4, channel_multiplier=8)

network7 = StylizedNet(bottleneck_size=3, channel_multiplier=32)
network8 = StylizedNet(bottleneck_size=3, channel_multiplier=16)
network9 = StylizedNet(bottleneck_size=3, channel_multiplier=8)

device_type = 'cpu'
device = torch.device(device_type)
model = network9.to(device)

summary(model, (3, 1024, 768), device=device_type)
