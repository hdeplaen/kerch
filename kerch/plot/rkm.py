import matplotlib.pyplot as plt

from ..rkm._kpca import _KPCA
from ..rkm._view import _View

def eigvals(kpca:_KPCA, log=True, ax=None) -> None:
    vals = kpca.vals.detach().cpu()

    indep = ax is None
    if indep:
        plt.figure()
        ax = plt.gca()

    ax.bar(range(len(vals)),vals)
    ax.set_title('Eigenvalues of ' + kpca.__str__())
    ax.set_xlabel('Different components')
    ax.set_ylabel('Eigenvalue magnitude')

    if log:
        ax.set_yscale('log')

    if indep:
        plt.show()

def correlation_hidden(view:_View, ax=None) -> None:
    h = view.hidden
    corr = (h.T @ h).detach().cpu()

    indep = ax is None
    if indep:
        plt.figure()
        ax = plt.gca()

    im = ax.imshow(corr)
    ax.set_title('Correlation of the hidden variables')
    plt.colorbar(im)

    if indep:
        plt.show()

