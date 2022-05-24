"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from abc import ABCMeta, abstractmethod

from .. import utils

class base(torch.nn.Module, metaclass=ABCMeta):
    r"""
    :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it will
        be given relative to these samples., defaults to `None`
    :param sample_trainable: `True` if the gradients of the sample points are to be computed. If so, a graph is
        computed and the sample can be updated. `False` just leads to a static computation., defaults to `False`
    :param center: `True` if any implicit feature or kernel is must be centered, `False` otherwise. The center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param num_sample: Number of sample points. This parameter is neglected if `sample` is not `None` and overwritten by
        the number of points contained in sample., defaults to 1
    :param dim_sample: Dimension of each sample point. This parameter is neglected if `sample` is not `None` and
        overwritten by the dimension of the sample points., defaults to 1

    :type sample: Tensor(num_sample, dim_sample), optional
    :type sample_trainable: bool, optional
    :type center: bool, optional
    :type num_sample: int, optional
    :type dim_sample: int, optional

    :param idx_sample: Initializes the indices of the samples to be updated. All indices are considered if both
        `idx_sample` and `prop_sample` are `None`., defaults to `None`
    :type idx_sample: int[], optional
    :param prop_sample: Instead of giving indices, specifying a proportion of the original sample set is also
        possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
        such that :math:`0 <` `prop_sample` :math:`\leq 1`. All indices are considered if both `idx_sample` and
        `prop_sample` are `None`., defaults to `None`.
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "sample": None,
        "sample_trainable": False,
        "center": False,
        "normalize": False,
        "num_sample": 1,
        "dim_sample": 1,
        "idx_sample": None,
        "prop_sample": None})
    def __init__(self, **kwargs):
        super(base, self).__init__()

        self.sample_trainable = kwargs["sample_trainable"]
        self._center = kwargs["center"]

        normalize = kwargs["normalize"]
        if normalize is True or normalize is False:
            self._eps = 1.e-8
            self._normalize = normalize
        else:
            self._eps = normalize
            self._normalize = True

        input_sample = kwargs["sample"]
        input_sample = utils.castf(input_sample)

        self._eps = 1.e-8


        if input_sample is not None:
            self._num_sample, self._dim_sample = input_sample.shape
        else:
            self._dim_sample = kwargs["dim_sample"]
            self._num_sample = kwargs["num_sample"]
        self.init_sample(input_sample, kwargs["idx_sample"], kwargs["prop_sample"])

    @abstractmethod
    def __str__(self):
        pass

    @property
    def params(self):
        r"""
        Dictionnary containing the parameters and their values. This can be relevant for monitoring.
        """
        return {}

    def _reset(self):
        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._K_norm = None
        self._phi = None
        self._C = None
        self._phi_mean = None
        self._phi_norm = None

    @property
    def dim_sample(self):
        r"""
        Dimension of each datapoint.
        """
        return self._dim_sample

    @property
    def num_sample(self):
        r"""
        Number of datapoints in the sample set.
        """
        return self._sample.shape[0]

    @property
    def num_idx(self):
        r"""
        Number of selected datapoints of the sample set when performaing various operations. This is only relevant in
        the case of stochastic training.
        """
        return len(self._idx_sample)

    @property
    def center(self) -> bool:
        r"""
        Indicates if the kernel has to be centered. Changing this value leads to a recomputation of the statistics.
        """
        return self._center

    @center.setter
    def center(self, val:bool):
        self._center = val
        self._reset()

    @property
    def normalize(self) -> bool:
        r"""
        Indicates if the kernel has to be normalized. Changing this value leads to a recomputation of the statistics.
        """
        return self._center

    @normalize.setter
    def normalize(self, val: bool):
        self._normalize = val
        self._reset()

    @property
    def idx(self):
        r"""
        Indices of the selected datapoints of the sample set when performing various operations. This is only relevant
        in the case of stochastic training.
        """
        return self._idx_sample

    @property
    def hparams(self):
        r"""
        Dictionnary containing the hyperparameters and their values. This can be relevant for monitoring.
        """
        return {"Trainable Kernels": self.sample_trainable,
                "center": self._center}

    @property
    def sample(self):
        r"""
        Sample dataset.
        """
        return self._sample.data

    def _all_sample(self):
        return range(self.num_sample)

    def train(self, mode=True):
        r"""
        Sets the kernel in training mode, which disables the gradients computation and disables stochasticity of the
        kernel. For the gradients and other things, we refer to the `torch.nn.Module` documentation. For the stachistic
        part, when put in evaluation mode (`False`), all the sample points are used for the computations, regardless of
        the previously specified indices.
        """
        if not mode:
            self.stochastic()
        return self

    def stochastic(self, idx_sample=None, prop_sample=None):
        """
        Resets which subset of the samples are to be used until the next call of this function. This is relevant in the
        case of stochastic training.

        :param idx_sample: Indices of the sample subset relative to the original sample set., defaults to `None`
        :type idx_sample: int[], optional
        :param prop_sample: Instead of giving indices, passing a proportion of the original sample set is also
            possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
            such that :math:`0 <` `prop_sample` :math:`\leq 1`., defaults to `None`.
        :type prop_sample: int, optional

        If `None` is specified for both `idx_sample` and `prop_sample`, all samples are used and the subset equals the
        original sample set. This is also the default behavior if this function is never called, nor the parameters
        specified during initialization.

        .. note::
            Both `idx_sample` and `prop_sample` cannot be filled together as conflict would arise.
        """
        self._reset()

        assert idx_sample is None or prop_sample is None, "Both idx_sample and prop_sample are not None. " \
                                                          "Please choose one non-None parameter only."

        if idx_sample is not None:
            self._idx_sample = idx_sample
        elif prop_sample is not None:
            assert prop_sample <= 1., 'Parameter prop_sample: the chosen proportion cannot be greater than 1.'
            assert prop_sample > 0., 'Parameter prop_sample: the chosen proportion must be strictly greater than 0.'
            n = self.num_sample
            k = torch.round(n * prop_sample)
            perm = torch.randperm(n)
            self._idx_sample = perm[:k]
        else:
            self._idx_sample = self._all_sample()

    def init_sample(self, sample=None, idx_sample=None, prop_sample=None):
        r"""
        Initializes the sample set (and the stochastic indices).

        :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it
            will be given relative to these samples. In case of overwriting a current sample, `num_sample` and
            `dim_sample` are also overwritten. If `None` is specified, the sample dataset will be initialized according
            to `num_sample` and `dim_sample` specified during the construction. If a previous sample set has been used,
            it will keep the same dimension by consequence., defaults to `None`
        :type sample: Tensor, optional
        :param idx_sample: Initializes the indices of the samples to be updated. All indices are considered if both
            `idx_sample` and `prop_sample` are `None`., defaults to `None`
        :type idx_sample: int[], optional
        :param prop_sample: Instead of giving indices, specifying a proportion of the original sample set is also
            possible. The indices will be uniformly randomly chosen without replacement. The value must be chosen
            such that :math:`0 <` `prop_sample` :math:`\leq 1`. All indices are considered if both `idx_sample` and
            `prop_sample` are `None`., defaults to `None`.
        """
        sample = utils.castf(sample)

        if sample is not None:
            self._num_sample, self._dim_sample = sample.shape
            self._sample = torch.nn.Parameter(sample.data,
                                        requires_grad=self.sample_trainable)
        else:
            self._sample = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty((self._num_sample, self._dim_sample), dtype=utils.FTYPE)),
                requires_grad=self.sample_trainable)

        self.stochastic(idx_sample, prop_sample)

    def update_sample(self, sample_values, idx_sample=None):
        r"""
        Updates the sample set. In contradiction to `init_samples`, this only updates the values of the sample and sets
        the gradients of the updated values to zero if relevant.

        :param sample_values: Values given to the updated samples.
        :param idx_sample: Indices of the samples to be updated. All indices are considered if `None`., defaults to
            `None`
        :type sample_values: Tensor
        :type idx_sample: int[], optional
        """
        sample_values = utils.castf(sample_values)
        self._reset()

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
    def _implicit(self, oos1=None, oos2=None):
        # implicit without center
        # explicit without center
        if oos1 is None:
            oos1 = self._sample[self._idx_sample, :]
        if oos2 is None:
            oos2 = self._sample[self._idx_sample, :]
        return oos1, oos2

    def _implicit_self(self, x=None):
        K = self.k(x,x, center=False, normalize=False)
        return torch.trace(K)

    def _compute_K_norm(self, center, x=None):
        r"""
        Computes the kernel norm given a sample.
        """
        if x is None and self._K_norm is None and center==self._center:
            self._K_norm = torch.sqrt(self._implicit_self())[:, None]
            return self._K_norm
        else:
            return torch.sqrt(self._implicit_self(x))[:, None]

    def _normalize_K(self, K, center, oos1=None, oos2=None):
        n_oos1 = self._compute_K_norm(center, oos1)
        n_oos2 = self._compute_K_norm(center, oos2).T
        N = n_oos1 * n_oos2
        return K / torch.clamp(N, min=1.e-8)

    @abstractmethod
    def _explicit(self, x=None):
        # explicit without center
        if x is None:
            x = self._sample[self._idx_sample, :]
        return x

    def _compute_K(self, implicit=False):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._K is None:
            if implicit:
                phi = self.phi()
                self._K = phi @ phi.T
            else:
                self._K = self._implicit()

                # centering in the implicit case happens ad hoc
                if self._center:
                    self._K_mean = torch.mean(self._K, dim=0, keepdim=True)
                    self._K_mean_tot = torch.mean(self._K, dim=(0, 1), keepdim=True)
                    self._K = self._K - self._K_mean \
                              - self._K_mean.T \
                              + self._K_mean_tot
                if self._normalize:
                    self._K = self._normalize_K(self._K)

        return self._K

    def _compute_C(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._C is None:
            self._phi = self._explicit()

            if self._center:
                self._phi_mean = torch.mean(self._phi, dim=0)
                self._phi = self._phi - self._phi_mean
            if self._normalize:
                self._phi_norm = torch.sqrt(torch.norm(self._phi, dim=1, keepdim=True))
                self._phi = self._phi / self._phi_norm
            self._C = self._phi.T @ self._phi
        return self._C, self._phi

    def phi(self, x=None, center=None, normalize=None):
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_sample), optional
        :param center: Returns if the matrix has to be centered or not. If None, then the default value used during
            construction is used., defaults to None
        :param normalize: Returns if the matrix has to be normalized or not. If None, then the default value used during
            construction is used., defaults to None
        :type center: bool, optional
        :type normalize: bool, optional
        :raises: PrimalError
        """

        # if x is None, phi(x) for x in the sample is returned.
        if center is None:
            center = self._center
        if normalize is None:
            normalize = self._normalize

        self._compute_C()

        x = utils.castf(x)
        phi = self._explicit(x)
        if center:
            phi = phi - self._phi_mean
        if normalize:
            phi = phi / torch.sqrt(torch.norm(phi, dim=1, keepdim=True))

        return phi

    def k(self, oos1=None, oos2=None, implicit=False, center=None, normalize=None):
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`oos1`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`oos2`).

        .. note::
            In the case of centered kernels, this computation is more expensive as it requires to center according to
            the sample dataset, which implies computing a statistic on the out-of-sample kernel matrix and thus
            also computing it.

        :param oos1: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :param oos2: Out-of-sample points (second dimension). If `None`, the default sample will be used., defaults to `None`

        :type oos1: Tensor(N,dim_sample), optional
        :type oos2: Tensor(M,dim_sample), optional

        :param center: Returns if the matrix has to be centered or not. If None, then the default value used during
            construction is used., defaults to None
        :param normalize: Returns if the matrix has to be normalized or not. If None, then the default value used during
            construction is used., defaults to None
        :type center: bool, optional
        :type normalize: bool, optional

        :return: Kernel matrix
        :rtype: Tensor(N,M)

        :raises: PrimalError
        """
        if center is None:
            center = self._center
        if normalize is None:
            normalize = self._normalize

        oos1 = utils.castf(oos1)
        oos2 = utils.castf(oos2)

        if implicit:
            self._compute_C()
            phi_sample = self.phi(oos2, center, normalize)
            phi_oos = self.phi(oos1, center, normalize)
            return phi_oos @ phi_sample.T
        else:
            self._compute_K()

        K = self._implicit(oos1, oos2)
        if center:
            if oos1 is not None:
                K_oos1_sample = self._implicit(oos1)
                m_oos1_sample = torch.mean(K_oos1_sample, dim=1, keepdim=True)
            else:
                m_oos1_sample = self._K_mean

            if oos2 is not None:
                K_oos2_sample = self._implicit(oos2)
                m_oos2_sample = torch.mean(K_oos2_sample, dim=1, keepdim=True)
            else:
                m_oos2_sample = self._K_mean

            K = K - m_oos1_sample \
                  - m_oos2_sample.T \
                  + self._K_mean_tot
        if normalize:
            K = self._normalize_K(center, oos1, oos2)
        return K

    def forward(self, x, representation="dual"):
        """
        Passes datapoints through the kernel.

        :param x: Datapoints to be passed through the kernel.
        :param representation: Chosen representation. If `dual`, an out-of-sample kernel matrix is returned. If
            `primal` is specified, it returns the explicit feature map., defaults to `dual`

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
        Returns the kernel matrix on the sample dataset. Same result as calling :py:func:`k()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.

        .. math::
            K_{ij} = k(x_i,x_j).
        """
        return self._compute_K()

    @property
    def C(self):
        r"""
        Returns the covariance matrix on the sample datapoints.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._compute_C()[0]

    @property
    def phi_sample(self):
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
        :py:func:`phi()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.
        """
        return self._compute_C()[1]
