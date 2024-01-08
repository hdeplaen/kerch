import torch
from torch.utils.data import SequentialSampler
import tqdm

from kerch.model._Model import _Model
from kerch._module._Logger import _Logger
from kerch.utils import kwargs_decorator
from kerch.opt import Optimizer
from kerch.data import _LearningSet


class Trainer(_Logger):
    @kwargs_decorator({'problem': 'regression',
                       'epochs': 100,
                       'batch_size': 'all',
                       'use_gpu': False,
                       'stiefel_lr': 1e-3,
                       'euclidean_lr': 1e-3,
                       'slow_lr': 1e-4})
    def __init__(self, model: _Model, learning_set: _LearningSet, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self._learning_set: _LearningSet = learning_set
        self._model: _Model = model
        self.problem = kwargs["problem"]
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.use_gpu = kwargs["use_gpu"]
        self.stiefel_lr = kwargs["stiefel_lr"]
        self.euclidean_lr = kwargs["euclidean_lr"]
        self.slow_lr = kwargs["slow_lr"]

    @property
    def problem(self) -> str:
        return self._problem

    @problem.setter
    def problem(self, val: str):
        msg = "The problem argument can either be set to the strings 'regression' or 'classification'."
        assert isinstance(val, str), msg
        if val == 'regression':
            self._loss = torch.nn.MSELoss(reduction='mean')
        elif val == 'classification':
            raise NotImplementedError
        else:
            raise AssertionError(msg)
        self._problem = val

    @property
    def model(self) -> _Model:
        return self._model

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, val: int):
        assert val > 0, "The number of epochs must be strictly positive."
        self._epochs = val

    @property
    def batch_size(self) -> int | str:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val: int | str):
        msg = "The batch_size argument can only take an integer or the string 'all' as valid option."
        num = len(self._learning_set.training_set)
        if isinstance(val, str):
            assert val == 'all', msg
            self._sampler = torch.utils.data.BatchSampler(SequentialSampler(range(num)),
                                                          batch_size=num, drop_last=False)
        else:
            assert isinstance(val, int), msg
            assert val > 0, \
                f"Only positive values for the batch size are allowed ({val} given)."
            assert val <= num, \
                f"The batch size ({val}) cannot be greater than the number of elements in the training set ({num})."
            self._sampler = torch.utils.data.BatchSampler(SequentialSampler(range(num)),
                                                          batch_size=val, drop_last=False)
        self._batch_size = val


    @property
    def _mask_batch_progress(self) -> bool:
        return self.batch_size == 'all'

    @property
    def _training_data(self) -> torch.Tensor:
        return (self._learning_set.training_set[:])[0]

    @property
    def _training_labels(self) -> torch.Tensor:
        return (self._learning_set.training_set[:])[1]

    @property
    def _validation_data(self) -> torch.Tensor:
        return (self._learning_set.validation_set[:])[0]

    @property
    def _validation_labels(self) -> torch.Tensor:
        return (self._learning_set.validation_set[:])[1]

    @property
    def _test_data(self) -> torch.Tensor:
        return (self._learning_set.test_set[:])[0]

    @property
    def _test_labels(self) -> torch.Tensor:
        return (self._learning_set.test_set[:])[1]

    @property
    def use_gpu(self) -> bool:
        r"""
        Specifies whether the gpu has to be used
        """
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, val: bool):
        self._use_gpu = val

    @property
    def stiefel_lr(self) -> float:
        r"""
        Learning rate for the Stiefel manifold
        """
        return self._stiefel_lr

    @stiefel_lr.setter
    def stiefel_lr(self, val: float):
        self._stiefel_lr = val

    @property
    def euclidean_lr(self) -> float:
        r"""
        Learning rate for the Euclidean manifold, i.e. the 'normal' parameters.
        """
        return self._euclidean_lr

    @euclidean_lr.setter
    def euclidean_lr(self, val: float):
        self._euclidean_lr = val

    @property
    def slow_lr(self) -> float:
        r"""
        Learning rate for specific hyperparameters than are better trained slower, i.e. the bandwidth of
        an RBF kernel.
        """
        return self._slow_lr

    @slow_lr.setter
    def slow_lr(self, val: float):
        self._slow_lr = val

    ############################################################################################################

    def _init_fit(self):
        if self.use_gpu:
            self.model.to('cuda:0')

        self.model.init_sample(self._training_data)
        self.model.init_levels()
        self.model.init_target(self._training_labels)

        self._optimizer = Optimizer(module=self.model,
                                    stiefel_lr=self._stiefel_lr,
                                    euclidean_lr=self._euclidean_lr,
                                    slow_lr=self._slow_lr)

    @torch.no_grad()
    def _problem_loss(self, data, labels) -> float:
        if len(data) == 0:
            return 0.
        self.model.eval()
        pred = self.model(data)
        self.model.train()
        return self._loss(pred, labels)


    def fit(self) -> _Model:
        self._init_fit()
        self.model.train()

        epoch_progress = tqdm.tqdm(range(self.epochs),
                                   position=0,
                                   desc="Epoch")
        batch_progress = tqdm.tqdm(self._sampler,
                                   leave=False,
                                   disable=self._mask_batch_progress,
                                   desc="Batch")

        def closure():
            self.model()
            loss = self._model.loss()
            if self._optimizer.requires_grad:
                loss.backward()
            return loss

        # TRAINING LOOP
        for epoch in epoch_progress:
            # preliminaries
            self._optimizer.zero_grad()
            if self.use_gpu:
                torch.cuda.empty_cache()
            # forward and backward
            for idx_batch in batch_progress:
                self.model.stochastic(idx=idx_batch)
                self._optimizer.step(closure)
                self.model.after_step()

            # validation and test
            train_loss = self._problem_loss(self._training_data, self._training_labels)
            val_loss = self._problem_loss(self._validation_data, self._validation_labels)
            test_loss = self._problem_loss(self._test_data, self._test_labels)

        self.model.eval()
        return self.model
