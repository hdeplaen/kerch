# coding=utf-8
from .decorators import (kwargs_decorator as kwargs_decorator, extend_docstring as extend_docstring)
from .cast import (castf as castf, casti as casti, check_representation as check_representation,
                   capitalize_only_first as capitalize_only_first)
from .type import (set_eps as set_eps, set_ftype as set_ftype, set_itype as set_itype, gpu_available as gpu_available,
                   FTYPE as FTYPE, ITYPE as ITYPE, EPS as EPS)
from .math import eigs as eigs
from .errors import (ImplicitError as ImplicitError,
                     ExplicitError as ExplicitError,
                     RepresentationError as RepresentationError,
                     BijectionError as BijectionError,
                     NotInitializedError as NotInitializedError,
                     MultiViewError as MultiViewError,
                     KerchError as KerchError)
from .tensor import (eye_like as eye_like, ones_like as ones_like, equal as equal)
from .defaults import (DEFAULT_KERNEL_TYPE as DEFAULT_KERNEL_TYPE,
                       DEFAULT_CACHE_LEVEL as DEFAULT_CACHE_LEVEL)