import argparse
from timeit import Timer


def measure_average_time(bottleneck_size, channel_multiplier, image_size, iterations, device_type='cuda'):
    setup = '''
gc.enable()

import time
import torch
from csfnst.fastneuralstyle.networks import TransformerNet

s = %s
m = %s
device_type = "%s"
device = torch.device(device_type)

model = TransformerNet(bottleneck_size=s, channel_multiplier=m, intermediate_activation_fn='PReLU', final_activation_fn='Sigmoid')
tensor = torch.randn((1, 3) + (%s, %s)).to(device)
model = model.to(device).eval()    

time.sleep(10)
    ''' % (bottleneck_size, channel_multiplier, device_type, image_size[0], image_size[1])

    code = 'model(tensor)'

    try:
        return Timer(
            stmt=code,
            setup=setup
        ).timeit(iterations) / float(iterations)
    except Exception as e:
        return -1


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bottleneck_size', required=True)
parser.add_argument('--channel_multiplier', required=True)
parser.add_argument('--width', required=True)
parser.add_argument('--height', required=True)
parser.add_argument('--iterations', default=10)
parser.add_argument('--device_type', default='cuda')

args = parser.parse_args()
result = measure_average_time(
    bottleneck_size=args.bottleneck_size,
    channel_multiplier=args.channel_multiplier,
    image_size=(args.width, args.height),
    iterations=args.iterations,
    device_type=args.device_type
)
print(
    f'bottlneck_size={args.bottleneck_size},'
    f' channel_multiplier={args.channel_multiplier},'
    f' image_size={args.width}x{args.height},'
    f' iterations={args.iterations},'
    f' device_type={args.device_type},'
    f' result={result}'
)
