# coding=utf-8
from __future__ import annotations

from .Watcher import Watcher


def factory(watcher_type:str='saver', **kwargs) -> Watcher | None:
    r"""
    # TODO
    """

    watcher = class_factory(watcher_type)
    return watcher(**kwargs)


def class_factory(watcher_type:str='saver'):
    if watcher_type == 'none':
        return None

    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    import kerch.monitor
    watcher_class = case_insensitive_getattr(kerch.watch, watcher_type)
    if watcher_class is None:
        raise NameError("Invalid watcher type.")
    return watcher_class

