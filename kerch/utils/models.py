import torch


def knn_weighted(coefficients: torch.Tensor, x: torch.Tensor, num: int = 1) -> torch.Tensor:
    r"""
    Returns a weighted sum of x by the coefficients.
    """
    num_points, num_coefficients = coefficients.shape
    num_x = x.shape[0]

    assert num_coefficients == num_x, \
        f'KNN: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points ({num_x}).'

    preimages = []
    for idx in range(num_points):
        sorted_coefficients, indices = torch.sort(coefficients[idx,:], descending=True)
        nearest_coefficients = sorted_coefficients[:num]

        normalized_coefficients = nearest_coefficients / torch.sum(nearest_coefficients)
        loc_sol = torch.einsum('i,ij->j', normalized_coefficients, x[indices[:num], :])
        preimages.append(loc_sol)
    return torch.vstack(preimages)


def knn(coefficients: torch.Tensor, x: torch.Tensor, num: int = 1) -> torch.Tensor:
    r"""
    Returns the sum of the num closest x, the distance being given by the coefficients.
    """
    num_points, num_coefficients = coefficients.shape[0]
    num_x = x.shape[0]

    assert num_coefficients == num_x, \
        f'KNN: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points ({num_x}).'

    preimages = []
    for idx in range(num_points):
        _, indices = torch.sort(coefficients, descending=True)
        loc_sol = torch.sum(x[indices[:num]], dim=0)
        preimages.append(loc_sol)
    return torch.vstack(preimages)
