import torch
from tqdm import trange
from torch import Tensor as T
from typing import List, Union

import kerch.utils
from .mvlevel import MVLevel
from kerch.rkm._kpca import _KPCA
from kerch import utils


class MVKPCA(_KPCA, MVLevel):
    r"""
    Multi-View Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(_KPCA)
    @utils.extend_docstring(MVLevel)
    def __init__(self, *args, **kwargs):
        super(MVKPCA, self).__init__(*args, **kwargs)

    def __str__(self):
        return "multi-view KPCA(" + MVLevel.__str__(self) + "\n)"


    def _project_primal(self, known, to_predict):
        phi_known = self.phi(known)
        weight_known = self.weights_by_name(list(known.keys()))
        weight_predict = self.weights_by_name(to_predict)

        Inv = torch.linalg.inv(torch.diag(self.vals) - weight_predict.T @ weight_predict)
        return phi_known @ weight_known @ Inv @ weight_predict.T


    def _project_dual(self, known, to_predict):
        k_known = self.k(known)
        K_predict = self.k(to_predict)

        Inv = torch.linalg.inv(torch.diag(self.vals) - self.H @ K_predict @ self.H.T)
        # return K_predict @ self.H.T @ Inv @ self.H @ k_known
        return k_known @ self.H.T @ Inv @ self.H @ K_predict

    def _project(self, known: dict, representation:str):
        r"""
        Predicts the feature map of the known not specified in the inputs, based on the values specified in the
        inputs.

        :param known: Dictionary of the inputs where the key is the view identifier (``str`` or ``int``) and the
            values the inputs to the known.
        :type known: dict
        :return:
        :rtype: Tensor
        """
        representation = utils.check_representation(representation, default=self._representation, cls=self)

        # CONSISTENCY
        num_points_known = None
        to_predict = []
        for key, _ in self.named_views:
            if key in known:
                value = known[key]
                # verify consistency of number of datapoints across the various provided inputs for the known.
                if value is not None:
                    if num_points_known is None:
                        num_points_known = value.shape[0]
                    else:
                        assert num_points_known == value.shape[0], \
                            f"Inconsistent number of datapoints to predict across the " \
                            f"different known: {num_points_known} and {value.shape[0]}."
            else:
                to_predict.append(key)
        assert num_points_known is not None, 'Nothing to predict.'

        # PREDICTION
        switcher = {'primal': self._project_primal,
                    'dual': self._project_dual}
        return switcher.get(representation, 'Error with the specified representation')(known, to_predict), to_predict

    def project(self, known:dict, representation:str=None) -> T:
        representation = utils.check_representation(representation, default=self._representation, cls=self)
        return self._project(known, representation)[0]

    def predict(self, known, representation=None, knn:int=1):
        representation = utils.check_representation(representation, default=self._representation, cls=self)
        projection, to_predict = self._project(known, representation)

        sol = {}
        if representation == 'primal':
            dim = 0
            for view, name in zip(self.views_by_name(to_predict), to_predict):
                view_phi = projection[:,dim:view.dim_feature]
                dim = view.dim_feature
                sol[name] = view.kernel.explicit_preimage(view_phi)
        elif representation == 'dual':
            for view, name in zip(self.views_by_name(to_predict), to_predict):
                sol[name] = view.kernel.implicit_preimage(projection, knn)

        return sol







    # def predict_opt(self, inputs: dict, representation='dual', lr: float = .001, tot_iter: int = 500) -> dict:
    #     # initiate parameters
    #     num_predict = None
    #     to_predict = []
    #     for key in self.views:
    #         if key in inputs:
    #             value = inputs[key]
    #             # verify consistency of number of datapoints across the various known.
    #             if num_predict is None:
    #                 num_predict = value.shape[0]
    #             else:
    #                 assert num_predict == value.shape[0], f"Inconsistent number of datapoints to predict across the " \
    #                                                       f"different known: {num_predict} and {value.shape[0]}."
    #         else:
    #             to_predict.append(key)
    #
    #     # if nothing is given, only one datapoint is predicted
    #     if num_predict is None:
    #         num_predict = 1
    #
    #     # initialize the other datapoints to be predicted
    #     params = torch.nn.ParameterList([])
    #
    #     def init_primal(params):
    #         for key in to_predict:
    #             v = self.view(key)
    #             inputs[key] = torch.nn.Parameter(
    #                 torch.zeros((num_predict, v.dim_input), dtype=utils.FTYPE),
    #                 requires_grad=True)
    #             params.append(inputs[key])
    #         return MVKPCA._primal_obj, params
    #
    #     def init_dual(params):
    #         for key in to_predict:
    #             v = self.view(key)
    #             inputs[key] = torch.nn.Parameter(
    #                 torch.zeros((num_predict, v.dim_input), dtype=utils.FTYPE),
    #                 requires_grad=True)
    #             params.append(inputs[key])
    #         return MVKPCA._dual_obj, params
    #
    #     switcher = {'primal': init_primal,
    #                 'dual': init_dual}
    #     if representation in switcher:
    #         fun, params = switcher.get(representation)(params)
    #     else:
    #         raise NameError('Invalid representation (must be primal or dual)')
    #
    #     # optimize
    #     bar = trange(tot_iter)
    #     opt = torch.optim.SGD(params, lr=lr)
    #     for _ in bar:
    #         opt.zero_grad()
    #         loss = fun(self, x=inputs)
    #         loss.backward(retain_graph=True)
    #         opt.step()
    #         bar.set_description(f"{loss:1.2e}")
    #
    #     return inputs
