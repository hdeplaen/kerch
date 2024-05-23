import torch

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
        loc_sol = torch.mean(x[indices[:num]], dim=0)
        preimages.append(loc_sol)
    return torch.vstack(preimages)