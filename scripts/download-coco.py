import os.path
from csfnst.utils import download_file, unzip

competition_year = '2017'

train_url = f'http://images.cocodataset.org/zips/train{competition_year}.zip'
val_url = f'http://images.cocodataset.org/zips/val{competition_year}.zip'
test_url = f'http://images.cocodataset.org/zips/test{competition_year}.zip'

train_save = '/home/s0555912/.datasets/coco/train.zip'
val_save = '/home/s0555912/.datasets/coco/val.zip'
test_save = '/home/s0555912/.datasets/coco/test.zip'

download_file(train_url, train_save)
unzip(train_save, os.path.dirname(train_save))

download_file(val_url, val_save)
unzip(val_save, os.path.dirname(val_save))

download_file(test_url, test_save)
unzip(test_save, os.path.dirname(test_save))
