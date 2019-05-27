import torch


def empty_tensor():
    """Returns an empty tensor."""

    return torch.tensor(0)


def is_empty_tensor(t):
    """Returns whether t is an empty tensor."""

    return len(t.size()) == 0
