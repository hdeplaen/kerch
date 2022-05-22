"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from .. import utils

class base(nn.Module, metaclass=ABCMeta):
    r"""
    :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it will
        be given relative to these samples., default to `None`
    :param sample_trainable: `True` if the gradients of the sample points are to be computed. If so, a graph is
        computed and the sample can be updated. `False` just leads to a static computation., defaults to `False`
    :param centering: `True` if any implicit feature or kernel is must be centered, `False` otherwise. The centering
        is always performed relative to a statistic on the sample., defaults to `False`
    :param num_sample: Number of sample points. This parameter is neglected if sample is not None and overwritten by
        the number of points contained in sample., defaults to 1
    :param dim_sample: Dimension of each sample point. This parameter is neglected if sample is not None and
        overwritten by the dimension of the sample points., defaults to 1

    :type sample: Tensor(num_sample, dim_sample), optional
    :type sample_trainable: bool, optional
    :type centering: bool, optional
    :type sample_num: int, optional
    :type sample_dim: int, optional
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "sample": None,
        "sample_trainable": False,
        "centering": False,
        "num_sample": 1,
        "dim_sample": 1})
    def __init__(self, **kwargs):
        super(base, self).__init__()

        self.sample_trainable = kwargs["sample_trainable"]
        self._centering = kwargs["centering"]

        input_sample = kwargs["sample"]
        input_sample = utils.castf(input_sample)

        if input_sample is not None:
            self._num_sample, self._dim_sample = input_sample.shape
        else:
            self._dim_sample = kwargs["dim_sample"]
            self._num_sample = kwargs["num_sample"]
        self.init_sample(input_sample)

        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._phi_mean = None
        self._idx_sample = None
        self.reset()

    @abstractmethod
    def __str__(self):
        pass

    @property
    def params(self):
        return {}

    def _empty_cache(self):
        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._phi_mean = None

    @property
    def num_sample(self):
        r"""
        Number of datapoints in the sample set.
        """
        return self._sample.shape[0]

    @property
    def num_idx(self):
        r"""
        Number of selected datapoints of the sample set when performaing various computations.
        """
        return len(self._idx_sample)

    @property
    def hparams(self):
        return {"Trainable Kernels": self.sample_trainable,
                "Centering": self._centering}

    @property
    def sample(self):
        r"""
        Sample dataset.
        """
        return self._sample.data

    def _all_sample(self):
        return range(self.num_sample())

    def train(self, mode=True):
        r"""
        Sets the kernel in training mode. Refer to torch.nn.Module documentation for more details. When put in
        evaluation mode (`False`), all the sample points are used for the computations, regardless of the
        previously specified indices.
        """
        if not mode:
            self.reset()
        return self

    def reset(self, idx_sample=None):
        """
        Resets which subset of the samples are to be used until the next call of this function.

        :param idx_sample: Indices of the sample subset relative to the original sample set. If `None` is specified,
            all samples are used and the subset equals the original sample set. This is also the default behavior if
            this function is never called., defaults to `None`
        :type idx_sample: int(), optional
        """
        self._empty_cache()
        if idx_sample is None:
            self._idx_sample = self._all_sample()
        else:
            self._idx_sample = idx_sample

    def init_sample(self, sample=None):
        """
        Initializes the sample set.

        :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it
            will be given relative to these samples. In case of overwriting a current sample, `num_sample` and
            `dim_sample` are also overwritten. If `None` is specified, the sample dataset will be initialized according
            to `num_sample` and `dim_sample` specified during the construction. If a previous sample set has been used,
            it will keep the same dimension by consequence., defaults to `None`
        :type sample: Tensor, optional
        """
        sample = utils.castf(sample)
        self._empty_cache()

        if sample is not None:
            self._num_sample, self._dim_sample = sample.shape
            self._sample = nn.Parameter(sample.data,
                                        requires_grad=self.sample_trainable)
        else:
            self._sample = nn.Parameter(
                nn.init.orthogonal_(torch.empty((self._num_sample, self._dim_sample), dtype=utils.FTYPE)),
                requires_grad=self.sample_trainable)

    def update_sample(self, sample_values, idx_sample=None):
        r"""
        Updates the sample set. In contradiction to `init_samples`, this only updates the values of the sample and sets
        the gradients of the updated values to zero if relevant.

        :param sample_values: Values given to the updated samples.
        :param idx_sample: Indices of the samples to be updated. All indices are considered if `None`., defaults to
            `None`
        :type sample_values: Tensor
        :type idx_sample: int(), optional
        """
        sample_values = utils.castf(sample_values)
        self._empty_cache()

        # use all indices if unspecified
        if idx_sample is None:
            idx_sample = self._all_sample()

        # check consistency of indices
        assert len(idx_sample) == sample_values.shape[0], f"Number of sample values ({sample_values.shape[0]}) and " \
                                                          f"corresponding indices ({len(idx_sample)}) are not " \
                                                          f"consistent."
        # update the values
        self._sample.data[idx_sample, :] = sample_values.data

        # zeroing relevant gradient if relevant
        if self.sample_trainable:
            self._sample.grad.data[idx_sample, :].zero_()

    # def merge_idxs(self, **kwargs):
    #     raise NotImplementedError
    #     # self.dmatrix()
    #     # return torch.nonzero(torch.triu(self.dmatrix()) > (1 - kwargs["mtol"]), as_tuple=False)
    #
    # def merge(self, idxs):
    #     raise NotImplementedError
    #     # # suppress added up kernel
    #     # self._sample = (self._sample.gather(dim=0, index=idxs[:, 1]) +
    #     #                 self._sample.gather(dim=0, index=idxs[:, 0])) / 2
    #
    #     self.dmatrix()
    #     # suppress added up kernel entries in the kernel matrix
    #     self._K.gather(dim=0, index=idxs[:, 1], out=self._K)
    #     self._K.gather(dim=1, index=idxs[:, 1], out=self._K)
    #
    # def reduce(self, idxs):
    #     raise NotImplementedError
    #     self._sample.gather(dim=0, index=idxs, out=self._sample)

###################################################################################################
################################### MATHS ARE HERE ################################################
###################################################################################################

    @abstractmethod
    def _implicit(self, x_oos=None, x_sample=None):
        # implicit without centering
        # explicit without centering
        if x_oos is None:
            x_oos = self._sample[self._idx_sample, :]
        if x_sample is None:
            x_sample = self._sample[self._idx_sample, :]
        return x_oos, x_sample

    @abstractmethod
    def _explicit(self, x=None):
        # explicit without centering
        if x is None:
            x = self._sample[self._idx_sample, :]
        return x

    def _dmatrix(self, implicit=False):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._K is None and not implicit:
            if self._idx_sample is None:
                self.reset()

            # self._k = self._implicit(self.kernels.gather(0, self._idx_kernels))
            self._K = self._implicit()

            if self._centering:
                n = self.num_sample()
                self._K_mean = torch.mean(self._K, dim=0)
                self._K_mean_tot = torch.mean(self._K, dim=(0, 1))
                self._K = self._K - self._K_mean.expand(n, n) \
                          - self._K_mean.expand(n, n).t() \
                          + self._K_mean_tot

        elif self._K is None and implicit:
            phi = self.phi()
            self._K = phi @ phi.T

        return self._K

    def _pmatrix(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._C is None:
            if self._idx_sample is None:
                self.reset()

            self._phi = self._explicit()

            if self._centering:
                self._phi_mean = torch.mean(self._phi, dim=0)
                self._phi = self._phi - self._phi_mean
            self._C = self._phi.T @ self._phi
        return self._C

    def phi(self, x=None):
        r"""
        Returns the explicit feature map.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            default to `None`
        :type x: Tensor(,dim_sample), optional
        :raises: PrimalError
        """

        # if x is None, phi(x) for x in the sample is returned.
        x = utils.castf(x)

        self._pmatrix()

        phi = self._explicit(x)
        if self._centering:
            phi = phi - self._phi_mean

        return phi

    def k(self, x_oos=None, x_sample=None, implicit=False):
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`x_oos`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`x_sample`).

        The case of centered kernels requires a particular discussion.

        :param x_oos: Out-of-sample points. If `None`, the default sample will be used., defaults to `None`
        :param x_sample: If `None`, the default sample is used. If not, the kernel matrix is computed relatively to
            another sample. This allows for a full out-of-sample matrix in both dimensions. It has no links with the
            original sample unless the statistic used for centering if relevant. If the kernel only exists in an
            implicit formulation and is centered, this will return an error as the centering consistent with the
            original sample is untractable then. One may use a Nyström kernel to force the existence of an explicit
            feature map., defaults to `None`

        :type x_oos: Tensor(N,dim_sample), optional
        :type x_sample: Tensor(M,dim_sample), optional

        :return K: Kernel matrix
        :rtype: Tensor(N,M)

        :raises: PrimalError
        """
        x_oos = utils.castf(x_oos)
        x_sample = utils.castf(x_sample)

        if x_oos is None and x_sample is None:
            return self._dmatrix(implicit=implicit)

        if x_sample is not None and not implicit and self._centering:
            raise NameError(
                "Impossible to compute centered out-of-sample to out-of-sample kernels for implicit-defined kernels as the centering statistic is only defined on the sample.")

        if implicit:
            self._pmatrix()
            phi_sample = self.phi(x_sample)
            phi_oos = self.phi(x_oos)
            Ky = phi_oos @ phi_sample.T
        else:
            self._dmatrix()
            Ky = self._implicit(x_oos, x_sample)
            if self._centering:
                Ky = Ky - torch.mean(Ky, dim=1, keepdim=True).expand(-1, Ky.shape[1])
                Ky = Ky - self._K_mean
                Ky = Ky + self._K_mean_tot

        return Ky

    def forward(self, x, representation="dual"):
        """
        Passes datapoints through the kernel.

        :param x: Datapoints to be passed through the kernel.
        :param representation: Chosen representation. If `dual`, an out-of-sample kernel matrix is returned. If
            `primal` is specified, it returns the explicit feature map., default to `dual`

        :type x: Tensor(,dim_sample)
        :type representation: str, optional

        :return: Out-of-sample kernel matrix or explicit feature map depending on `representation`.

        :raises: RepresentationError
        """

        def primal(x):
            return self.phi(x)

        def dual(x):
            return self.k(x)

        switcher = {"primal": primal,
                    "dual": dual}

        fun = switcher.get(representation, utils.model.RepresentationError)
        return fun(x)

    @property
    def K(self):
        r"""
        Returns the kernel matrix on the sample. Same result as calling :py:func:`k()`, but faster as no assertions
        and tests have to be performed. It is loaded from memory if already computed and unchanged since then, to avoid
        re-computation when reccurently called.

        .. math::
            K_{ij} = k(x_i,x_j).
        """
        return self._dmatrix()

    @property
    def C(self):
        r"""
        Returns the covariance matrix.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._pmatrix()[0]

    @property
    def phi_sample(self):
        r"""
        Returns the explicit feature map of the sample datapoints. Same as calling :py:func:`phi()`, but faster as no
        assertions or tests have to be performed. It is loaded from memory if already computed and unchanged since
        then, to avoid re-computation when reccurently called.
        """
        return self._pmatrix()[1]