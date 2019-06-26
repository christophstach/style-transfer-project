from time import time

import torch.cuda
from torchsummary import summary

from csfnst.fastneuralstyle.networks import CustomTransformerNetInterpolate, TransformerNet, BottleneckType
from csfnst.utils import load_image

image_size = 400
force_cpu = True
content_image_file = '../images/content/brad_pitt_01.jpg'

device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
content_image = load_image(content_image_file, size=(image_size, image_size)).to(device)

n1 = CustomTransformerNetInterpolate(channel_multiplier=32).to(device)
n2 = TransformerNet(channel_multiplier=16, bottleneck_size=5, bottleneck_type=BottleneckType.MOBILE_VERSION_TWO_BLOCK,
                    expansion_factor=6).to(device)
n1.eval()
n2.eval()


# summary(n1, (3, image_size, image_size), device='cpu')
#summary(n2, (3, image_size, image_size), device='cpu')

print('\n######### Running Networks #########\n')

start = time()
output_image_n1 = n1(content_image.unsqueeze(0)).squeeze()
end = time()
print(f'Time elapsed calculating the new image by Net1: {(end - start):,.4f} sec')

start = time()
output_image_n2 = n2(content_image.unsqueeze(0)).squeeze()
end = time()
print(f'Time elapsed calculating the new image by Net2: {(end - start):,.4f} sec')

# plot_image_tensor(output_image_n1, show=True)
# plot_image_tensor(output_image_n2, show=True)

print(output_image_n1.shape)
print(output_image_n2.shape)

