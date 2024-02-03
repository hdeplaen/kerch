# coding=utf-8

# ABSTRACT
from ._base_kernel import _BaseKernel
from .kernel import Kernel as Kernel
from .implicit import Implicit as Implicit
from .explicit import Explicit as Explicit
from .distance.distance import Distance as Distance
from .distance.distance_squared import DistanceSquared as DistanceSquared
from .distance.select_distance import SelectDistance as SelectDistance

# GENERIC
from .generic.linear import Linear as Linear
from .generic.rbf import RBF as RBF
from .generic.rff import RFF as RFF
from .generic.laplacian import Laplacian as Laplacian
from .generic.cosine import Cosine as Cosine
from .generic.sigmoid import Sigmoid as Sigmoid
from .generic.polynomial import Polynomial as Polynomial
from .generic.random_features import RandomFeatures as RandomFeatures

# NETWORK
from .network.explicit_nn import ExplicitNN as ExplicitNN
from .network.implicit_nn import ImplicitNN as ImplicitNN

# TIME
from .time.indicator import Indicator as Indicator
from .time.hat import Hat as Hat

# VISION
from .vision.additive_chi_2 import AdditiveChi2 as AdditiveChi2
from .vision.skewed_chi_2 import SkewedChi2 as SkewedChi2

# STATISTICS
from .statistics.epanechnikov import Epanechnikov as Epanechnikov
from .statistics.uniform import Uniform as Uniform
from .statistics.triangular import Triangular as Triangular
from .statistics.quartic import Quartic as Quartic
from .statistics.triweight import Triweight as Triweight
from .statistics.tricube import Tricube as Tricube
from .statistics.logistic import Logistic as Logistic
from .statistics.silverman import Silverman as Silverman
from .statistics.exponential import Exponential as Exponential

# MISC
from .nystrom import Nystrom as Nystrom
from ._factory import factory as factory