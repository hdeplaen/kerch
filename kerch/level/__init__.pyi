# coding=utf-8
from .single_view import (KPCA as KPCA,
                          LSSVM as LSSVM,
                          PPCA as PPCA,
                          Ridge as Ridge)
from .multi_view import (MVKPCA as MVKPCA)
from .factory import factory as factory