# coding=utf-8
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
import socket
import os
import torch

from ..opt import Optimizer
from ..model.Model import Model


class Watcher(metaclass=ABCMeta):
    r"""
    :param model: Model to watch (monitor, log and save)
    :param opt: Optimizer used in the training
    :param expe_name: Name of the experiment
    :param verbose: Verbosity boolean. Defaults to False.
    :param num_epochs_loss: The losses will be logged every `num_epochs_loss`. Defaults to 1.
    :param num_epochs_save: The model will be saved every `num_epochs_save`. Defaults to 1.
    :param num_epochs_params: The model parameters will be logged every `num_epochs_params`. Defaults to 1.
    :param num_epochs_plot: The watched properties will be logged/plotted every `num_epochs_plot`. Defaults to 1.
    :type model: kerch.model.Model
    :type opt: kerch.opt.Optimizer
    :type expe_name: str
    :type verbose: bool, optional
    :type num_epochs_loss: int, optional
    :type num_epochs_save: int, optional
    :type num_epochs_params: int, optional
    :type num_epochs_plot: int, optional
    """

    def __init__(self, model: Model, opt: Optimizer | None, expe_name: str, verbose: bool = False, **kwargs):
        super(Watcher, self).__init__(**kwargs)
        self._verbose = verbose
        self._num_epochs_loss = kwargs.pop('num_epochs_loss', 1)
        self._num_epochs_params = kwargs.pop('num_epochs_params', 1)
        self._num_epochs_save = kwargs.pop('num_epochs_save', 1)
        self._num_epochs_plot = kwargs.pop('num_epochs_plot', 1)

        # MODEL
        self._model = model
        assert isinstance(model, Model), 'The model argument is not a correct Kerch Model.'

        # DIRECTORY
        current_time = datetime.now().strftime('%b%d_%_H-%M-%S')
        self._id = current_time + '_' + socket.gethostname()
        assert isinstance(expe_name, str), 'The expe_name argument must be a string.'
        self._expe_name = expe_name
        self._expe_id = self._model.__class__.__name__ + self._id
        self._dir_project = os.path.join("kerch", self._expe_name, self._expe_id)
        self._dir_model = os.path.join(self._dir_project, "model")

        # OPTIMIZER
        if opt is not None:
            assert isinstance(opt, Optimizer), 'The opt argument is not a correct Kerch Optimizer.'
        self._opt = opt

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def model(self) -> Model:
        r"""
        Model used for logging purposes
        """
        return self._model

    @property
    def opt(self) -> Optimizer:
        r"""
        Optimizer used for logging purposes
        """
        return self._opt

    @property
    def expe_name(self) -> str:
        r"""
        Experiment name
        """
        return self._expe_name

    @property
    def expe_id(self) -> str:
        r"""
        Unique experiment identifier
        """
        return self._expe_id

    @property
    def dir_project(self) -> str:
        r"""
        Relative path of the project directory.
        """
        return self._dir_project

    def save_model(self, epoch: int | None = None) -> str:
        r"""
        Saves the model current state.

        :param epoch: Current epoch number. If None, it is considered to be the final state. Defaults to None.
        :type epoch: int, optional
        :return: The relative path where the model is saved
        :rtype: str
        """
        if not os.path.exists(self._dir_model):
            os.makedirs(self._dir_model)

        save_name = 'final' if epoch is None else 'epoch-' + str(epoch)
        save_path = os.path.join(self._dir_model, save_name + '.pt')
        torch.save(self._model.state_dict(), save_path)
        return save_path

    def finish(self) -> str:
        r"""
        Cleans all the checkpoints of the model states, but the final one.

        :return: The relative path where the final model has been saved.
        :rtype: str
        """
        final_name = 'final.pt'
        filelist = [f for f in os.listdir(self._dir_model) if f != final_name]
        for f in filelist:
            os.remove(os.path.join(self._dir_model, f))
        return os.path.join(self._dir_model, final_name)

    @abstractmethod
    def update(self,
               epoch: int,
               objective_loss: float,
               training_error: float | None = None,
               validation_error: float | None = None,
               test_error: float | None = None
               ) -> None:
        pass
