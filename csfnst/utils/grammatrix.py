import torch


def gram_matrix(tensor, normalize=True):
    """
    Flattens width and high dimension to a flattened vector but keeps the channel dimension.
    The result is a 2D matrix, which is multiplied with it's own transposed version.
    Also can calculate batches of images.

    :param tensor:
    :param normalize:
    :return:
    """

    assert len(tensor.shape) == 3 or len(tensor.shape) == 4, 'Can only calculate gram matrix of 3D tensors or 4D'

    if len(tensor.shape) == 3:
        c, h, w = tensor.shape
        normalizer = c * h * w
        tensor_flat = tensor.flatten(1)

        if normalize:
            return torch.div(
                torch.matmul(
                    tensor_flat,
                    tensor_flat.t()
                ),
                normalizer
            )
        else:
            return torch.matmul(
                tensor_flat,
                tensor_flat.t()
            )

    if len(tensor.shape) == 4:
        b, c, h, w = tensor.shape
        normalizer = c * h * w
        tensor_flat = tensor.flatten(2)

        if normalize:
            return torch.div(
                torch.bmm(
                    tensor_flat,
                    tensor_flat.transpose(1, 2)
                ),
                normalizer
            )
        else:
            return torch.bmm(
                tensor_flat,
                tensor_flat.transpose(1, 2)
            )
