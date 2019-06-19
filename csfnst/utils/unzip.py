import zipfile


def unzip(file, save_location):
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall(save_location)
    zip_ref.close()
