from timeit import Timer

from prettytable import PrettyTable


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

size = (640, 480)
print('###################################')
print('640*480')
print('###################################')
t = PrettyTable(['Name', 'CPU', 'GPU'])
t.add_row([
    'Network 1 (s=5, m=32)',
    round(measure_average_time(5, 32, 'experiment__net1__s5__m32__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(5, 32, 'experiment__net1__s5__m32__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 2 (s=5, m=16)',
    round(measure_average_time(5, 16, 'experiment__net2__s5__m16__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(5, 16, 'experiment__net2__s5__m16__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 3 (s=5, m= 8)',
    round(measure_average_time(5, 8, 'experiment__net3__s5__m8__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(5, 8, 'experiment__net3__s5__m8__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 4 (s=4, m=32)',
    round(measure_average_time(4, 32, 'experiment__net4__s4__m32__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(4, 32, 'experiment__net4__s4__m32__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 5 (s=4, m=16)',
    round(measure_average_time(4, 16, 'experiment__net5__s4__m16__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(4, 16, 'experiment__net5__s4__m16__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 6 (s=4, m= 8)',
    round(measure_average_time(4, 8, 'experiment__net6__s4__m8__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(4, 8, 'experiment__net6__s4__m8__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 7 (s=3, m=32)',
    round(measure_average_time(3, 32, 'experiment__net7__s3__m32__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(3, 32, 'experiment__net7__s3__m32__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 8 (s=3, m=16)',
    round(measure_average_time(3, 16, 'experiment__net8__s3__m16__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(3, 16, 'experiment__net8__s3__m16__residual_block.pth', size, iterations, 'cuda'), 5)
])
t.add_row([
    'Network 9 (s=3, m= 8)',
    round(measure_average_time(3, 8, 'experiment__net9__s3__m8__residual_block.pth', size, iterations, 'cpu'), 5),
    round(measure_average_time(3, 8, 'experiment__net9__s3__m8__residual_block.pth', size, iterations, 'cuda'), 5)
])
print(t)
