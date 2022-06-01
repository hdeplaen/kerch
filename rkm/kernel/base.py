"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from abc import ABCMeta, abstractmethod
from torch import Tensor

from .. import utils

class base(torch.nn.Module, metaclass=ABCMeta):
    r"""
    :param sample: Sample points used to compute the kernel matrix. When an out-of-sample computation is asked, it will
        be given relative to these samples., defaults to `None`
    :param _sample_trainable: `True` if the gradients of the sample points are to be computed. If so, a graph is
        computed and the sample can be updated. `False` just leads to a static computation., defaults to `False`
    :param center: `True` if any implicit feature or kernel is must be centered, `False` otherwise. The center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param normalize: `True` if any implicit feature or kernel is must be normalized, `False` otherwise. The center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param num_sample: Number of sample points. This parameter is neglected if `sample` is not `None` and overwritten by
        the number of points contained in sample., defaults to 1
    :param dim_input: Dimension of each sample point. This parameter is neglected if `sample` is not `None` and
        overwritten by the dimension of the sample points., defaults to 1

    :type sample: Tensor(num_sample, dim_input), optional
    :type _sample_trainable: bool, optional
    :type center: bool, optional
    :type normalize: bool, optional
    :type num_sample: int, optional
    :type dim_input: int, optional

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
        "_sample_trainable": False,
        "center": False,
        "normalize": False,
        "num_sample": None,
        "dim_input": None,
        "idx_sample": None,
        "prop_sample": None})
    def __init__(self, **kwargs):
        super(base, self).__init__()
        utils.logger.debug("Initializing new kernel.")

        self._sample = torch.nn.Parameter(torch.empty((0,0)))
        self._sample_trainable = kwargs["_sample_trainable"]
        self._center = kwargs["center"]

        # normalization settings
        normalize = kwargs["normalize"]
        if normalize is True or normalize is False:
            self._eps = 1.e-8
            self._normalize_requested = normalize
        else:
            self._eps = normalize
            self._normalize_requested = True

        # It may be that some kernels are naturally normalized and don't need the additional computation
        self._normalize = self._normalize_requested

        sample = kwargs["sample"]
        sample = utils.castf(sample)

        self._eps = 1.e-8

        if sample is not None:
            self._num_sample, self._dim_input = sample.shape
        else:
            self._dim_input = kwargs["dim_input"]
            self._num_sample = kwargs["num_sample"]

        self.init_sample(sample, kwargs["idx_sample"], kwargs["prop_sample"])

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    @property
    def params(self):
        r"""
        Dictionnary containing the parameters and their values. This can be relevant for monitoring.
        """
        return {}

    def _apply(self, fn):
        with torch.no_grad():
            for _, cache_entry in self._cache.items():
                cache_entry.data = fn(cache_entry)
        return super(base, self)._apply(fn)

    def _reset(self):
        self._cache = {}

    @property
    def dim_input(self) -> int:
        r"""
        Dimension of each datapoint.
        """
        return self._dim_input

    @dim_input.setter
    def dim_input(self, val:int):
        assert self._dim_input is None, "Cannot set the dimension of the sample points after initialization if the " \
                                        "sample dataset. Use init_sample() instead."
        self._dim_input = val
        if self._num_sample is not None:
            self.init_sample()

    @property
    def num_sample(self) -> int:
        r"""
        Number of datapoints in the sample set.
        """
        return self._num_sample

    @num_sample.setter
    def num_sample(self, val:int):
        assert self._num_sample is None, "Cannot set the dimension of the sample points after initialization if the " \
                                       "sample dataset. Use init_sample() instead."
        self._num_sample = val
        if self._dim_input is not None:
            self.init_sample()

    @property
    def num_idx(self) -> int:
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
        return self._normalize

    @normalize.setter
    def normalize(self, val: bool):
        self._normalize = val
        self._reset()

    @property
    def idx(self) -> Tensor:
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
        return {"Trainable Kernels": self._sample_trainable,
                "center": self._center}

    @property
    def sample(self) -> Tensor:
        r"""
        Sample dataset.
        """
        return self._sample.data

    @sample.setter
    def sample(self, val):
        assert val is not None, 'The assigned sample cannot be None.'
        self.init_sample(val)

    @property
    def sample_as_param(self):
        return self._sample

    @property
    def sample_trainable(self) -> bool:
        r"""
        Boolean if the sample dataset can be trained.
        """
        return self._sample_trainable

    @sample_trainable.setter
    def sample_trainable(self, val: bool):
        self._sample_trainable = val
        self._sample.requires_grad = self._sample_trainable

    @property
    def current_sample(self) -> Tensor:
        r"""
        Returns the sample that is currently used in the computations and for the normalizing and centering statistics
        if relevant.
        """
        return self._sample[self._idx_sample, :]

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
        :type prop_sample: double, optional

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
            `dim_input` are also overwritten. If `None` is specified, the sample dataset will be initialized according
            to `num_sample` and `dim_input` specified during the construction. If a previous sample set has been used,
            it will keep the same dimension by consequence. A last case occurs when `sample` is of the class
            `torch.nn.Parameter`: the sample will then use those values and they can thus be shared with the module
            calling this method., defaults to `None`
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

        if sample is None:
            utils.logger.debug("Initializing new sample with the sample dimensions.")
            if self._num_sample is None and self.dim_input is None:
                utils.logger.info('The sample  cannot be initialized because no sample dataset has been provided nor the '
                             'sample dimensions have been initialized yet.')
                return
            self._sample = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty((self._num_sample, self._dim_input),
                                                      dtype=utils.FTYPE,
                                                      device=self._sample.device),),
                requires_grad=self._sample_trainable)
        elif isinstance(sample, torch.nn.Parameter):
            utils.logger.debug("Using existing sample defined as an external parameter.")
            self._num_sample, self._dim_input = sample.shape
            self._sample = sample
        else:
            utils.logger.debug("Assigning new sample based on the given values.")
            self._num_sample, self._dim_input = sample.shape
            self._sample = torch.nn.Parameter(sample.data,
                                              requires_grad=self._sample_trainable)

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

        if self._sample.nelement() == 0:
            utils.logger.warning('Cannot update the sample values of a None sample dataset.')
            return

        sample_values = utils.castf(sample_values, dev=self._sample.device)
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
        if self._sample_trainable:
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
    #     self._cache["K"].gather(dim=0, index=idxs[:, 1], out=self._cache["K"])
    #     self._cache["K"].gather(dim=1, index=idxs[:, 1], out=self._cache["K"])
    #
    # def reduce(self, idxs):
    #     raise NotImplementedError
    #     self._sample.gather(dim=0, index=idxs, out=self._sample)

    ###################################################################################################
    ################################### MATHS ARE HERE ################################################
    ###################################################################################################

    @abstractmethod
    def _implicit(self, x=None, y=None):
        # implicit without center
        if x is None:
            x = self.current_sample
        if y is None:
            y = self.current_sample
        return x, y

    def _implicit_self(self, x=None):
        K = self._implicit(x,x)
        return torch.diag(K)

    @abstractmethod
    def _explicit(self, x=None):
        # explicit without center
        if x is None:
            x = self.current_sample
        return x

    def _compute_K(self, implicit=False):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._sample.nelement() == 0:
            utils.logger.warning('No sample dataset. Please assign a sample dataset or specify the dimensions of the '
                            'sample dataset to initialize random values before computing kernel values.')
            return None

        if "K" not in self._cache:
            utils.logger.debug("Computing kernel matrix and dual statistics.")
            if implicit:
                phi = self.phi()
                self._cache["K"] = phi @ phi.T
            else:
                self._cache["K"] = self._implicit()

                # centering in the implicit case happens ad hoc
                if self._center:
                    self._cache["K_mean"] = torch.mean(self._cache["K"], dim=1, keepdim=True)
                    self._cache["K_mean_tot"] = torch.mean(self._cache["K"], dim=(0, 1))
                    self._cache["K"] = self._cache["K"] - self._cache["K_mean"] \
                              - self._cache["K_mean"].T \
                              + self._cache["K_mean_tot"]
                if self._normalize:
                    self._cache["K_norm"] = torch.sqrt(torch.diag(self._cache["K"]))[:,None]
                    K_norm = self._cache["K_norm"] * self._cache["K_norm"].T
                    self._cache["K"] = self._cache["K"] / torch.clamp(K_norm, min=self._eps)

        return self._cache["K"]

    def _compute_C(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._sample.nelement() == 0:
            utils.logger.warning('No sample dataset. Please specify a sample dataset or the dimensions of the sample '
                            'dataset to initialize random values before computing kernel values.')
            return None

        if "C" not in self._cache:
            utils.logger.debug("Computing explicit feature map and primal statistics.")
            self._cache["phi"] = self._explicit()

            if self._center:
                self._cache["phi_mean"] = torch.mean(self._cache["phi"], dim=0)
                self._cache["phi"] = self._cache["phi"] - self._cache["phi_mean"]
            if self._normalize:
                self._cache["phi_norm"] = torch.norm(self._cache["phi"], dim=1, keepdim=True)
                self._cache["phi"] = self._cache["phi"] / self._cache["phi_norm"]
            self._cache["C"] = self._cache["phi"].T @ self._cache["phi"]
        return self._cache["C"], self._cache["phi"]

    def phi(self, x=None, center=None, normalize=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
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

        if self._compute_C() is None:
            return None

        x = utils.castf(x)
        phi = self._explicit(x)
        if center:
            phi = phi - self._cache["phi_mean"]
        if normalize:
            phi_norm = torch.norm(phi, dim=1, keepdim=True)
            phi = phi / torch.clamp(phi_norm, min=self._eps)

        return phi

    def k(self, x=None, y=None, implicit=False, center=None, normalize=None) -> Tensor:
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`x`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`y`).

        .. note::
            In the case of centered kernels, this computation is more expensive as it requires to center according to
            the sample dataset, which implies computing a statistic on the out-of-sample kernel matrix and thus
            also computing it.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :param y: Out-of-sample points (second dimension). If `None`, the default sample will be used., defaults to `None`

        :type x: Tensor(N,dim_input), optional
        :type y: Tensor(M,dim_input), optional

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
        # if x is None and y is None:
        #     return self.K

        if center is None:
            center = self._center
        if normalize is None:
            normalize = self._normalize

        x = utils.castf(x)
        y = utils.castf(y)

        if implicit:
            if self._compute_C() is None:
                return None
            phi_sample = self.phi(y, center, normalize)
            phi_oos = self.phi(x, center, normalize)
            return phi_oos @ phi_sample.T
        else:
            if self._compute_K() is None:
                return None

        K = self._implicit(x, y)
        if center:
            if x is not None:
                K_x_sample = self._implicit(x)
                m_x_sample = torch.mean(K_x_sample, dim=1, keepdim=True)
            else:
                m_x_sample = self._cache["K_mean"]

            if y is not None:
                K_y_sample = self._implicit(y)
                m_y_sample = torch.mean(K_y_sample, dim=1, keepdim=True)
            else:
                m_y_sample = self._cache["K_mean"]

            K = K - m_x_sample \
                  - m_y_sample.T \
                  + self._cache["K_mean_tot"] 
        if normalize:
            if x is None:
                n_x = self._cache["K_norm"]
            else:
                diag_K_x = self._implicit_self(x)[:, None]
                if center:
                    diag_K_x = diag_K_x - 2 * m_x_sample + self._cache["K_mean_tot"] 
                n_x = torch.sqrt(diag_K_x)

            if y is None:
                n_y = self._cache["K_norm"]
            else:
                diag_K_y = self._implicit_self(y)[:, None]
                if center:
                    diag_K_y = diag_K_y - 2 * m_y_sample + self._cache["K_mean_tot"] 
                n_y = torch.sqrt(diag_K_y)

            K_norm = n_x * n_y.T
            K = K / torch.clamp(K_norm, min=self._eps)

        return K

    def forward(self, x, representation="dual") -> Tensor:
        """
        Passes datapoints through the kernel.

        :param x: Datapoints to be passed through the kernel.
        :param representation: Chosen representation. If `dual`, an out-of-sample kernel matrix is returned. If
            `primal` is specified, it returns the explicit feature map., defaults to `dual`

        :type x: Tensor(,dim_input)
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
    def K(self) -> Tensor:
        r"""
        Returns the kernel matrix on the sample dataset. Same result as calling :py:func:`k()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.

        .. math::
            K_{ij} = k(x_i,x_j).
        """
        return self._compute_K()

    @property
    def C(self) -> Tensor:
        r"""
        Returns the covariance matrix on the sample datapoints.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._compute_C()[0]

    @property
    def phi_sample(self) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
        :py:func:`phi()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.
        """
        return self._compute_C()[1]
