import torch


def log(content, style='-'):
    """Logging with underline."""

    print(content)
    print(style * len(content.strip()))


def empty_tensor():
    """Returns an empty tensor."""

    return torch.tensor(0)


def is_empty_tensor(t):
    """Returns whether t is an empty tensor."""

    return len(t.size()) == 0
