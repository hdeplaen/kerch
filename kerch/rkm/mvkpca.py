import torch
from tqdm import trange
from torch import Tensor as T

from .mvlevel import MVLevel
from kerch import utils


class MVKPCA(MVLevel):
    r"""
    Multi-View Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(MVLevel)
    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(MVKPCA, self).__init__(*args, **kwargs)
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)

    def __repr__(self):
        return "Multi-View KPCA" + super(MVKPCA, self).__repr__()

    @property
    def vals(self) -> T:
        return self._vals.data

    @vals.setter
    def vals(self, val):
        val = utils.castf(val, tensor=False, dev=self._vals.device)
        self._vals.data = val

    ###################################################################

    def _solve_dual(self) -> None:
        if self.dim_output is None:
            self._dim_output = self.num_idx

        K = self.K
        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        self.hidden = h
        self.vals = v

    def _solve_primal(self) -> None:
        if self.dim_output is None:
            self._dim_output = self.dim_input

        C = self.C
        v, w = utils.eigs(C, k=self.dim_output, psd=True)

        self.weight = w
        self.vals = v

    ####################################################################

    def _primal_obj(self, x=None) -> T:
        P = self.weight @ self.weight.T  # primal projector
        R = self._I_primal - P  # reconstruction
        C = self.c(x)  # covariance
        return torch.norm(R * C)  # reconstruction error on the covariance

    def _dual_obj(self, x=None) -> T:
        P = self.hidden @ self.hidden.T  # dual projector
        R = self._I_dual - P  # reconstruction
        K = self.k(x)  # kernel matrix
        return torch.norm(R * K)  # reconstruction error on the kernel

    ####################################################################

    def generate(self, h=None):
        if h is None:
            raise NotImplementedError
        raise NotImplementedError

    def predict_proj(self, inputs: dict, method='closed'):
        r"""
        Predicts the feature map of the views not specified in the inputs, based on the values specified in the
        inputs.

        The closed form method uses

        .. math::
            \Psi = \Phi U V^\top \left( I - VV^\top\right)^{-1},


        whereas the fixed point iteration uses

        .. math::
            H_{k+1} = \Phi U + \Psi_k V,
            \Psi_{k+1} = H_{k+1} V^\top.


        :param inputs: Dictionnary of the inputs to be used for the prediction.
        :param method: Type of method used: ``'closed'`` for the closed-form solution or ``'iter'`` for the fixed
            point iteration., defaults to ``'closed'``
        :type inputs: dict
        :type method: str, optional
        :return: Predictions for the views not specified in the inputs.
        :rtype: Tensor (will become a dict in a later version)
        """
        num_predict = None
        to_predict = []
        for key in self.views:
            if key in inputs:
                value = inputs[key]
                # verify consistency of number of datapoints across the various views.
                if num_predict is None:
                    num_predict = value.shape[0]
                else:
                    assert num_predict == value.shape[0], f"Inconsistent number of datapoints to predict across the " \
                                                          f"different views: {num_predict} and {value.shape[0]}."
            else:
                to_predict.append(key)

        assert num_predict is not None, 'Nothing to predict.'

        def _closed_form(u, v, phi):
            self._log.debug('Using the closed form prediction. Faster but potentially unstable if components assigned '
                            'with too small eigenvalues are used.')
            Proj = v @ v.T
            Inv = torch.linalg.pinv(utils.eye_like(Proj) - Proj)
            return phi @ u @ v.T @ Inv

        def _iter_fixed(u, v, phi):
            self._log.debug(
                '[BETA]. Using the fixed point iteration prediction scheme. This is more stable than the closed '
                'form solution, but may be very slow. Please prefer the closed form method if no '
                'components of too small values are used.')
            h_update = lambda psi: (phi @ u + psi @ v)
            psi_update = lambda h: h @ v.T

            dim = 0
            for key in to_predict:
                dim += self.view(key).dim_feature

            psi = torch.zeros((num_predict, dim), dtype=utils.FTYPE)
            for _ in range(1000):
                h = h_update(psi)
                psi = psi_update(h)
            return psi

        phi = self.phi(inputs)
        u = self.weight_from_views(list(inputs.keys()))
        v = self.weight_from_views(to_predict)

        switcher = {'closed': _closed_form,
                    'iter': _iter_fixed}
        psi = switcher.get(method, "Unknown method")(u, v, phi)
        # TODO: format sol in dictionary (and change doc)
        return psi

    def predict_opt(self, inputs: dict, representation='dual', lr: float = .001, tot_iter: int = 500) -> dict:
        # initiate parameters
        num_predict = None
        to_predict = []
        for key in self.views:
            if key in inputs:
                value = inputs[key]
                # verify consistency of number of datapoints across the various views.
                if num_predict is None:
                    num_predict = value.shape[0]
                else:
                    assert num_predict == value.shape[0], f"Inconsistent number of datapoints to predict across the " \
                                                          f"different views: {num_predict} and {value.shape[0]}."
            else:
                to_predict.append(key)

        # if nothing is given, only one datapoint is predicted
        if num_predict is None:
            num_predict = 1

        # initialize the other datapoints to be predicted
        params = torch.nn.ParameterList([])

        def init_primal(params):
            for key in to_predict:
                v = self.view(key)
                inputs[key] = torch.nn.Parameter(
                    torch.zeros((num_predict, v.dim_input), dtype=utils.FTYPE),
                    requires_grad=True)
                params.append(inputs[key])
            return MVKPCA._primal_obj, params

        def init_dual(params):
            for key in to_predict:
                v = self.view(key)
                inputs[key] = torch.nn.Parameter(
                    torch.zeros((num_predict, v.dim_input), dtype=utils.FTYPE),
                    requires_grad=True)
                params.append(inputs[key])
            return MVKPCA._dual_obj, params

        switcher = {'primal': init_primal,
                    'dual': init_dual}
        if representation in switcher:
            fun, params = switcher.get(representation)(params)
        else:
            raise NameError('Invalid representation (must be primal or dual)')

        # optimize
        bar = trange(tot_iter)
        opt = torch.optim.SGD(params, lr=lr)
        for _ in bar:
            opt.zero_grad()
            loss = fun(self, x=inputs)
            loss.backward(retain_graph=True)
            opt.step()
            bar.set_description(f"{loss:1.2e}")

        return inputs
