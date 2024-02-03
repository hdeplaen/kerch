"""
Implementation of an exponential kernel based on the L1 distance. You could inherit from implicit and
defining an _implicit function. However, in the specific case of an exponential kernel, you can just redefine
a specific _dist function. This allows for an automatic management of the bandwidth. For more details, we refer
the documentation of the exponential kernel class (in comparison to the explicit class).

Author: HENRI DE PLAEN
Date: June 2022
"""

import torch
from kerch.kernel.generic.exponential import exponential


class exp_l1(exponential):
    # the two following methods are sufficient to define the exponential

    def __init__(self, **kwargs):
        # as the bandwidth is managed by the exponential kernel (abstract) mother class,
        # you don't need to define anything.
        super(exp_l1, self).__init__(**kwargs)

    def _dist(self, x, y):
        # x and y will never be None and always have the shape of [num_x, dim_x] and [num_y, dim_y]
        # with dim_x = dim_y, but not necessarily num_x = num_y.

        x = x.T[:, :, None]     # dimension broadcasting leading to [dim_x, num_x, :]
        y = y.T[:, None, :]     # dimension broadcasting leading to [dim_y, :, num_y]

        diff = x - y            # the dimensions are now [dim_x=dim_y, num_x, num_y]
        return torch.sum(torch.abs(diff), dim=0, keepdim=False) # [num_x, num_y]

    # two beautify the kernel, you can also add the following methods.
    # if not added, default values will be returned.

    def __str__(self):
        # if sigma is not defined and you want to print the kernel, you will get an error.
        # you can therefore verify before is it exists (if it is defined by a heuristic as
        # provided by the exponential kernel (abstract) mother class.
        try:
            return f"_Exponential-L1 kernel (sigma: {self.sigma})"
        except AttributeError:
            return f"_Exponential-L1 kernel (sigma undefined)"

    @property
    def hparams(self):
        # this is to be used to get back a lot of parameters and info typically for monitoring
        # with tensorboard, weights&biases or other.
        return {"Kernel": "_Exponential-L1",
                **super(exp_l1, self).hparams_fixed}