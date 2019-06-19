import os.path
from urllib.request import urlretrieve

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_file(url, save_location, replace=False):
    directory = os.path.dirname(save_location)

    if not replace and os.path.exists(save_location):
        print(f'File {save_location} already exists, skipping...')
        return

    if not os.path.exists(directory):
        os.makedirs(directory)

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, unit_divisor=1024, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=save_location, reporthook=t.update_to, data=None)
