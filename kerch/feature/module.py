# coding=utf-8
"""
Abstract class defining a general level in the toolbox.
"""
from __future__ import annotations

import torch
from typing import Iterator
from abc import ABCMeta, abstractmethod

from .logger import Logger
from .. import _GLOBALS
from ..utils import capitalize_only_first, extend_docstring


@extend_docstring(Logger)
class Module(Logger,
             torch.nn.Module,
             object,
             metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # for some obscure reason, calling the super init does not lead to the call of both classes.
        # by consequence, we make the calls manually to each parents
        torch.nn.Module.__init__(self)
        Logger.__init__(self, *args, **kwargs)

    def __repr__(self):
        return capitalize_only_first(self.__str__())


    @property
    @extend_docstring(Logger.logging_level)
    def logging_level(self) -> int:
        return self._logger_internal.level

    @logging_level.setter
    def logging_level(self, level: int | None):
        if level is None:
            level = _GLOBALS["LOG_LEVEL"]
        self._logger_internal.setLevel(level)
        for child in self.children():
            if isinstance(child, Logger):
                child.logging_level = level

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        r"""
        Iterator yielding all parameters lying on the Euclidean manifold (standard optimization). The optimization is
        done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.

        :param recurse: If ``True``, yields both the Euclidean parameters of this module and its potential children.
            otherwise, it only yields Euclidean parameters from this module. Defaults to ``True``.
        :type recurse: bool, optional
        :return: Euclidean parameters
        :rtype: Iterator[torch.nn.Parameter]
        """
        if recurse:
            for module in self.children():
                if isinstance(module, Module):
                    yield from module._euclidean_parameters(recurse)

    def _stiefel_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        r"""
        Iterator yielding all parameters that must lie on the Stiefel manifold (optimization is done onto that manifold).
        The Stiefel manifold corresponds to the orthonormal parameters :math:`U \in \mathrm{St}(n,m)`, i.e., all
        :math:`U \in \mathbb{R}^{n \times m}` such that :math:`U^\top U = I`. The dimensions :math:`n` and :math:`m` are
        proper to each parameter.

        :param recurse: If ``True``, yields both the Stiefel parameters of this module and its potential children.
            otherwise, it only yields Stiefel parameters from this module. Defaults to ``True``.
        :type recurse: bool, optional
        :return: Stiefel parameters
        :rtype: Iterator[torch.nn.Parameter]
        """
        if recurse:
            for module in self.children():
                if isinstance(module, Module):
                    yield from module._stiefel_parameters(recurse)

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        r"""
        Iterator yielding all parameters lying on the Euclidean manifold (standard optimization). The optimization is
        done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.

        The specificity of these slow Euclidean parameters is that they are better trained with a lower learning rate that the
        others, hence their name and the necessity to group them apart.

        :param recurse: If ``True``, yields both the slow (Euclidean) parameters of this module and its potential children.
            otherwise, it only yields slow (Euclidean) parameters from this module. Defaults to ``True``.
        :type recurse: bool, optional
        :return: Slow (Euclidean) parameters
        :rtype: Iterator[torch.nn.Parameter]
        """
        if recurse:
            for module in self.children():
                if isinstance(module, Module):
                    yield from module._slow_parameters(recurse)

    def manifold_parameters(self, recurse=True, type='euclidean') -> Iterator[torch.nn.Parameter]:
        r"""
        Iterator yielding the parameters of a specific type. A distinction is made between three types:

        * ``'euclidean'``:
            parameters lying on the Euclidean manifold (standard optimization). The optimization is
            done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.
        * ``'stiefel'``:
            parameters that must lie on the Stiefel manifold (optimization is done onto that manifold).
            The Stiefel manifold corresponds to the orthonormal parameters :math:`U \in \mathrm{St}(n,m)`, i.e., all
            :math:`U \in \mathbb{R}^{n \times m}` such that :math:`U^\top U = I`. The dimensions :math:`n` and :math:`m` are
            proper to each parameter.
        * ``'slow'``:
            parameters lying on the Euclidean manifold (standard optimization). The optimization is
            done onto :math:`\mathbb{R}^{n \times m}`, :math:`n` and :math:`m` depending on the size of each parameter.
            The specificity of these slow Euclidean parameters is that they are better trained with a lower learning rate that the
            others, hence their name and the necessity to group them apart.

        We refer to the documentation of :doc:`../features/module` for further information.

        :param type: Which parameters group the method must return. The three values above are accepted. Defaults to ``'euclidean'``.
        :type type: str, optional
        :param recurse: If ``True``, yields both the specified parameters of this module and its potential children.
            otherwise, it only yields the specified parameters from this module. Defaults to ``True``.
        :type recurse: bool, optional
        :return: Parameters of the specified type
        :rtype: Iterator[torch.nn.Parameter]
        """
        switcher = {'euclidean': self._euclidean_parameters,
                    'stiefel': self._stiefel_parameters,
                    'slow': self._slow_parameters}
        gen = switcher.get(type, 'Invalid manifold name.')

        memo = set()
        for p in gen(recurse=recurse):
            if p not in memo:
                memo.add(p)
                yield p

    def before_step(self) -> None:
        r"""
        Specific operations to be performed before a training step. We refer to the documentation of
        :doc:`../features/module` for further information.
        """
        pass

    def after_step(self) -> None:
        r"""
            Specific operations to be performed after a training step. We refer to the documentation of
            :doc:`../features/module` for further information.
            """
        pass

    @property
    def hparams_variable(self) -> dict:
        r"""
        Variable hyperparameters of the module. By contrast with :py:attr:`hparams_fixed`, these are the values that are may change during
        the training and may be monitored at various stages during the training.
        If applicable, these can be kernel bandwidth parameters for example.

        .. note::

            All parameters that are potentially trainable, like a kernel bandwidth :math:`\sigma` for example, are
            included in this dictionary, even if the corresponding trainable argument is set to ``False``. In the
            latter case, they will be not evolve during training iterations, but will still be included in this
            dictionary.

        We refer to the documentation of :doc:`../features/module` for further information.
        """
        return {}

    @property
    def hparams_fixed(self) -> dict:
        r"""
        Fixed hyperparameters of the module. By contrast with :py:attr:`hparams_variable`, these are the values that are fixed and
        cannot possibly change during the training. If applicable, these can be specific architecture values for example.
        We refer to the documentation of :doc:`../features/module` for further information.
        """
        return {}
