import torch


def label_propagate(X, y_l, threshold=0.95):
    """Perform label propagation with similiarity graph.

    Arguments:
        X: input tensor of size (n_l + n_u, d), where n_l is number of labeled samples,
            n_u is number of unlabeled samples and d is the dimension of input
        y_l: label tensor of size (n_l, c), where c is the number of classes
        threshold: similarity threshold for label propagation

    Returns:
        y_u: propagated label tensor of size (n_u, c)
    """

    # disable gradient computation
    X = X.detach()
    y_l = y_l.detach()

    # number of labeled and unlabeled samples
    n_l = y_l.size(0)
    n_u = X.size(0) - n_l

    # compute similarity matrix W
    Xp = X.view(X.size(0), 1, X.size(1))
    W = torch.exp(-torch.einsum('ijk, ijk->ij', X - Xp, X - Xp))

    # sub-matrix of W containing similarities between labeled and unlabeled samples
    W_ul = W[n_l:, :n_l]

    # max_similarities is the maximum similarity for each unlabeled sample
    # src_indexes is the respective labeled sample index
    max_similarities, src_indexes = W_ul.max(dim=1)

    # initialize y_u with zeros
    y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

    # only propagate labels if maximum similarity is above the threhold
    propagated_samples = max_similarities > threshold
    y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

    return y_u
