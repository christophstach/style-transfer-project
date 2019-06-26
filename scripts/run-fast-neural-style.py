from datetime import datetime

from csfnst.fastneuralstyle import run_fast_neural_style

run_fast_neural_style(
    model_path='../checkpoints/transformer_net_up_sample_conv_transpose__starry_night_google__128.pth',
    image_size=128,
    content_image_file='../images/content/hidden/adrian_01.jpg',
    output_image_file=f'../images/output/hidden/adrian_01_{datetime.now()}.jpg',
    force_cpu=True
)
