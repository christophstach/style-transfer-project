from timeit import Timer
import argparse


def measure_average_time(s, m, checkpoint_name, image_size, runs=1, device_type='cuda'):
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
    ''' % (s, m, device_type, image_size[0], image_size[1])

    code = 'model(tensor)'

    try:
        return Timer(
            stmt=code,
            setup=setup
        ).timeit(runs) / float(runs)
    except Exception as e:
        return -1


iterations = 10

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                   const=sum, default=max,
                   help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
