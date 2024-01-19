# coding=utf-8
from __future__ import annotations

from abc import ABCMeta, abstractmethod
import os

from .Watcher import Watcher

class Plotter(Watcher, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(Plotter, self).__init__(*args, **kwargs)
        if self._plotter_name is None:
            self._dir_plotter = self.dir_project
        else:
            self._dir_plotter = os.path.join(self._dir_project, self._plotter_name)

    @property
    @abstractmethod
    def _plotter_name(self) -> str | None:
        pass

    @property
    def dir_plotter(self) -> str:
        r"""
        Relative path of the plotter directory.
        """
        return self._dir_plotter