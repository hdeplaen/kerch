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

    def predict(self, inputs: dict, representation='dual', lr: float = .001, tot_iter: int = 500) -> dict:
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

    # def _predict_dual(self, inputs: dict, lr: float = .001, tot_iter: int = 50) -> dict:
    #     assert isinstance(inputs, dict), "The input must be a dictionary containing the values to predict."
    #
    #     from matplotlib import pyplot as plt
    #
    #     K_base = 0.
    #     num_predict = None
    #     to_do = []
    #
    #     # the kernel matrices of the provided input are first computed.
    #     for key in self.views:
    #         if key in inputs:
    #             value = inputs[key]
    #             # verify consistency of number of datapoints across the various views.
    #             if num_predict is None:
    #                 num_predict = value.shape[0]
    #             else:
    #                 assert num_predict == value.shape[0], f"Inconsistent number of datapoints to predict across the " \
    #                                                       f"different views: {num_predict} and {value.shape[0]}."
    #             K_base += self.view(key).k(x=value)
    #         else:
    #             to_do.append(key)
    #
    #     if num_predict is None:
    #         self._log.warning('Nothing to predict.')
    #         return {}
    #
    #     # # get projector and reconstruction error
    #     dev = K_base.device
    #     P = self.hidden @ self.hidden.T
    #     I = torch.eye(self.num_idx, self.num_idx, dtype=utils.FTYPE, device=dev)
    #     R = I - P
    #
    #     # set init values (in the idea of continuation, starting from the closest points in the dataset)
    #     # center K_base
    #     K_min = K_base - torch.mean(K_base, dim=0, keepdim=True) \
    #             - torch.mean(K_base, dim=1, keepdim=True)
    #     idx = torch.argmax(K_min, dim=1)
    #
    #     params = {}
    #     for key in to_do:
    #         v = self.view(key)
    #         # x = torch.zeros(num_predict, v.dim_input, dtype=utils.FTYPE, device=dev, requires_grad=True)
    #         x = torch.tensor(v.sample[idx, :], dtype=utils.FTYPE, device=dev, requires_grad=True)
    #         params[key] = x
    #
    #     plt.figure(5)
    #     plt.plot(inputs['time'], x.data)
    #     plt.show()
    #
    #     # update K based on the current parameters values
    #     def update_K(K_base, params):
    #         K = K_base.clone()
    #         for key, p in params.items():
    #             K += self.view(key).k(p)
    #         return K
    #
    #     # debug
    #     x = torch.linspace(-5, 5, 100)
    #     KT = K_base.T[:, None, :]
    #     KX = self.view('space').k(x).T[:, :, None]
    #     K = KT + KX
    #     PK = torch.einsum('ij,ikl->jkl', R, K)  # (s, mx, mt)
    #     f = torch.sum(PK ** 2, dim=0)  # (mx, mt)
    #     f = f / torch.norm(f, dim=0)
    #     plt.imshow(torch.flipud(torch.log(f)))
    #     plt.show()
    #     idx_min = torch.argmin(f, dim=0)
    #     plt.plot(params['space'].data)
    #     plt.plot(x[idx_min])
    #     plt.show()
    #
    #     # optimize
    #     bar = trange(tot_iter)
    #     for _ in bar:
    #         K = update_K(K_base, params)  # (m,n)
    #         RK = K @ R  # (m,n)
    #         del K
    #         loss = torch.sum(RK ** 2)
    #         loss.backward(retain_graph=True)
    #         bar.set_description(f"{loss:1.2e}")
    #         with torch.no_grad():
    #             for _, p in params.items():
    #                 p.sub_(p.grad, alpha=lr)
    #                 p.grad.zero_()
    #
    #     return params
