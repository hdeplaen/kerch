from abc import ABCMeta
from .sigma import Sigma
from ...utils import extend_docstring


@extend_docstring(Sigma)
class Select(Sigma, metaclass=ABCMeta):
    r"""
    :param distance: The normed used. Defaults to ``'euclidean'```. Other options are ``'manhattan'``, ``'chebyshev'``,
        ``'minkowski'`` and ``'wasserstein'``.
    :type distance: str
    """

    def __new__(cls, *args, **kwargs):
        distance = kwargs.pop('distance', 'euclidean')
        if isinstance(distance, str):
            distance = distance.lower()
            if distance == 'euclidean':
                from .euclidean import Euclidean
                distance_class = Euclidean
            elif distance == 'chebyshev':
                from .chebyshev import Chebyshev
                distance_class = Chebyshev
            elif distance == 'manhattan':
                from .manhattan import Manhattan
                distance_class = Manhattan
            elif distance == 'wasserstein':
                from .wasserstein import Wasserstein
                distance_class = Wasserstein
            elif distance == 'minkowski':
                from .minkowski import Minkowski
                distance_class = Minkowski
            else:
                raise ValueError('Unrecognized distance')
        elif isinstance(distance, Sigma):
            distance_class = distance
        else:
            raise ValueError('The distance must either be a string or a class inheriting from '
                             'kerch.kernel.distance.Sigma.')
        new_cls = type(cls.__name__, (cls, distance_class,), dict(cls.__dict__))
        return Sigma.__new__(new_cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(Select, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"({super(Select, self).__str__()}, sigma: {str(self.sigma)})"
        return f"({super(Select, self).__str__()}, sigma undefined)"
