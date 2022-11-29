import torch
from tqdm import trange
from torch import Tensor as T
from typing import List, Union

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

    ## MATH

    def _phi_predict(self, weight_predict: T, weight_known: T, phi_known: T) -> T:
        r"""
        .. math::
            \Psi = \Phi U \left( I - VV^\top\right)^{-1} V,

        where :math:`\Phi` are the concatenated feature maps of the different given input views (N x sum dims),
        :math:`U` the corresponding weights (sum dims x s) and :math:`V` the weights of the views to be predicted
        (sum dims x s).
        """
        # TECHNIQUE 1
        # weight_predict_pinv = torch.pinverse(weight_predict)
        # WL = (weight_known @ torch.diag(self.vals)).unsqueeze(0)
        # phi_phi_weight_known = torch.einsum('ni,nj,jk->nik', phi_known, phi_known, weight_known)
        # norm_known = torch.einsum('ni,ni->n', phi_known, phi_known)
        # predicted_unnormed = torch.einsum('ni,nik,kl->nl',phi_known,WL-phi_phi_weight_known,weight_predict_pinv)
        # return predicted_unnormed / norm_known.unsqueeze(1)

        # TECNHINQUE 2
        # vals_sqrt = torch.sqrt(self.vals)
        # vals_sqrt_inv = torch.diag(1 / vals_sqrt)
        # vals_sqrt_id = torch.diag(vals_sqrt)
        # Proj = vals_sqrt_id @ weight_predict.T @ weight_predict @ vals_sqrt_inv
        # Recon = torch.linalg.inv(utils.eye_like(Proj) - Proj)
        # return phi_known @ weight_known @ vals_sqrt_inv @ Recon @ vals_sqrt_id @ weight_predict.T

        # TECNHIQUE 3
        weight_known_norm = torch.sum((weight_known ** 2), dim=0, keepdim=True)
        weight_known_normed = weight_known / weight_known_norm
        return phi_known @ weight_known_normed @ weight_predict.T

        # Proj = weight_predict.T @ weight_predict
        # Recon = torch.linalg.inv(utils.eye_like(Proj) - Proj)
        # return phi_known @ weight_known @ Recon @ weight_predict.T

    ##########################################################################

    def predict_oos(self, known: dict) -> T:
        r"""
        Predicts the feature map of the views not specified in the inputs, based on the values specified in the
        inputs.

        :param known: Dictionary of the inputs where the key is the view identifier (``str`` or ``int``) and the
            values the inputs to the views.
        :type known: dict
        :return:
        :rtype: Tensor
        """
        # CONSISTENCY
        num_points_known = None
        to_predict = []
        for key, _ in self.named_views:
            if key in known:
                value = known[key]
                # verify consistency of number of datapoints across the various provided inputs for the views.
                if num_points_known is None:
                    num_points_known = value.shape[0]
                else:
                    assert num_points_known == value.shape[0], \
                        f"Inconsistent number of datapoints to predict across the " \
                        f"different views: {num_points_known} and {value.shape[0]}."
            else:
                to_predict.append(key)
        assert num_points_known is not None, 'Nothing to predict.'

        # PREDICTION
        phi_known = self.phi(known)
        weight_known = self.weights_by_name(list(known.keys()))
        weight_predict = self.weights_by_name(to_predict)
        return self._phi_predict(weight_predict, weight_known, phi_known)

    def predict_sample(self, names: Union[str, List[str]]) -> T:
        r"""
        Predicts the values explicit feature map of the views in the names list, based on the other views,
        not mentioned.

        :param names: Names of the views to be predicted, based on the non-listed ones.
        :type names: List[str]
        """
        assert self._representation == 'primal', utils.PrimalError
        # construct two lists:
        #   known: the views not in name that are serving as base,
        #   unknown: the views that are to be predicted (names).
        if isinstance(names, str):
            names = [names]
        known = []
        for key, _ in self.named_views:
            if key not in names:
                known.append(key)

        phi_known = self.phi(known)
        weight_known = self.weights_by_name(known)
        weight_predict = self.weights_by_name(names)
        # return self._phi_predict(weight_predict, weight_known, phi_known)
        return self._phi_predict(weight_predict, weight_known, phi_known)

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
