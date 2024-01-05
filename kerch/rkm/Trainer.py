import torch
from torch.utils.data import DataLoader
import tqdm

from .RKM import RKM
from .._Logger import _Logger
from ..utils import kwargs_decorator, castf
from ..opt import Optimizer


class Trainer(_Logger):
    @kwargs_decorator({'problem': 'regression',
                       'train_data': None,
                       'train_labels': None,
                       'test_data': None,
                       'test_labels': None,
                       'epochs': 100,
                       'batch_size': 'all',
                       'shuffle': True,
                       'use_gpu': False,
                       'stiefel_lr': 1e-3,
                       'euclidean_lr': 1e-3,
                       'slow_lr': 1e-4})
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self._model: RKM = kwargs["model"]
        self.problem = kwargs["problem"]
        self.batch_size = kwargs["batch_size"]
        self.shuffle = kwargs["shuffle"]
        self.epochs = kwargs["epochs"]

        self.train_data = kwargs["train_data"]
        self.train_labels = kwargs["train_labels"]
        self.test_data = kwargs["test_data"]
        self.test_labels = kwargs["test_labels"]

        self.use_gpu = kwargs["use_gpu"]
        self.stiefel_lr = kwargs["stiefel_lr"]
        self.euclidean_lr = kwargs["euclidean_lr"]
        self.slow_lr = kwargs["slow_lr"]

    @property
    def problem(self) -> str:
        return self._problem

    @problem.setter
    def problem(self, val: str):
        msg = "The problem argument can iether be set to the strings 'regression' or 'classification'."
        assert isinstance(val, str), msg
        if val == 'regression':
            self._loss = torch.nn.MSELoss(reduction='mean')
        elif val == 'classification':
            raise NotImplementedError
        else:
            raise AssertionError(msg)
        self._problem = val

    @property
    def model(self) -> RKM:
        return self._model

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, val: int):
        assert val > 0, "The number of epochs must be strictly positive."
        self._epochs = val

    @property
    def shuffle(self) -> bool:
        r"""
        True if the batch are to be shuffled within the data or False if taken in the order of the data
        """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, val: bool):
        self._shuffle = val

    @property
    def batch_size(self) -> int | str:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val: int | str):
        msg = "The batch_size argument can only take an integer or the string 'all' as valid option."
        if isinstance(val, str):
            assert val == 'all', msg

        else:
            assert isinstance(val, int), msg
            assert val > 0, 'Only positive values for the batch size are allowed'
        self._batch_size = val

    @property
    def train_data(self) -> torch.Tensor | None:
        return self._train_data.dataset

    @train_data.setter
    def train_data(self, val):
        val = castf(val)
        if val is None:
            self._train_data = None
            return
        batch_size = self.batch_size
        if batch_size == 'all':
            batch_size = val.shape[0]
        self._train_data = DataLoader(val,
                                      batch_size=batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=1)

    @property
    def num_train(self) -> int | None:
        if self._train_data is None:
            return None
        return len(self._train_data.dataset)

    @property
    def train_labels(self) -> torch.Tensor | None:
        return self._train_labels

    @train_labels.setter
    def train_labels(self, val):
        val = castf(val)
        if val is not None:
            assert self.num_train is not None, 'Please assign the training labels first.'
            assert val.shape[0] == self.num_train, \
                (f"The number of training labels ({val.shape[0]}) does not correspond to the number of labels given "
                 f"({self.num_train})")
            self._train_labels = val

    @property
    def test_data(self) -> torch.Tensor | None:
        return self._test_data

    @test_data.setter
    def test_data(self, val):
        self._test_data = castf(val)

    @property
    def num_test(self) -> int | None:
        if self._test_data is None:
            return None
        return self._test_data.shape[0]

    @property
    def test_labels(self) -> torch.Tensor | None:
        return self._test_labels

    @test_labels.setter
    def test_labels(self, val):
        val = castf(val)
        if val is not None:
            assert self.num_test is not None, 'Please assign the testing labels first.'
            assert val.shape[0] == self.num_test, \
                (f"The number of testing labels ({val.shape[0]}) does not correspond to the number of labels given "
                 f"({self.num_test})")
            self._test_labels = val

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

        assert self.train_data is not None, 'Please provide a training data before training the model.'
        self.model.init_sample(self._train_data)
        self.model.init_levels()
        if self.train_labels is not None:
            self.model.init_targets(self.train_labels)

        self._optimizer = Optimizer(mdl=self.model,
                                    stiefel_lr=self._stiefel_lr,
                                    euclidean_lr=self._euclidean_lr,
                                    slow_lr=self._slow_lr)

    def _problem_loss(self, data, labels) -> float:
        self.model.eval()
        pred = self.model(data)
        return self._loss(pred, labels)

    def fit(self) -> RKM:
        self._init_fit()
        self.model.train()

        epoch_progress = tqdm.tqdm(range(self.epochs))
        batch_progress = tqdm.tqdm()

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
            for batch in batch_progress:
                self._optimizer.step(closure)
                for level in self._model.levels:
                    level.after_step()

            # validation and test
            # val_loss = self._problem_loss()
            test_loss = self._problem_loss(self.test_data, self.test_labels)

        self.model.eval()
        return self.model
