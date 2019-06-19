import torch


def three_dim(dim):
    dims = torch.stack((dim, dim, dim))
    return dims
