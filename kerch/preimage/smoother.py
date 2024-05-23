import torch


def smoother(coefficients: torch.Tensor, x: torch.Tensor, num: int = 1) -> torch.Tensor:
    r"""
    Returns a weighted sum of x by the coefficients.
    """
    num_points, num_coefficients = coefficients.shape
    num_x = x.shape[0]

    assert num_coefficients == num_x, \
        f'KNN: Incorrect number of coefficients ({num_coefficients}), ' \
        f'compared to the number of points ({num_x}).'

    if num >= num_coefficients:
        return torch.einsum('ni,ij->nj', coefficients / torch.sum(coefficients, dim=0), x)

    preimages = []
    for idx in range(num_points):
        sorted_coefficients, indices = torch.sort(coefficients[idx, :], descending=True)
        nearest_coefficients = sorted_coefficients[:num]

        normalized_coefficients = nearest_coefficients / torch.sum(nearest_coefficients)
        loc_sol = torch.einsum('i,ij->j', normalized_coefficients, x[indices[:num], :])
        preimages.append(loc_sol)
    return torch.vstack(preimages)
