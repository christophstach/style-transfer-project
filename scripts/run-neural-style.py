from datetime import datetime

from csfnst.neuralstyle import run_neural_style

run_neural_style(
    content_image_file='../images/content/hoovertowernight_01.jpg',
    style_image_file='../images/style/candy_01.jpg',
    output_image_file=f'../images/output/hoovertowernight_01_{datetime.now()}.jpg'
)
