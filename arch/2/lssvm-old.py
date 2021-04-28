import numpy as np
from sklearn import metrics

# TO DO : create a LS-SVM object
class lssvm():
    @staticmethod
    def dual(input, targets, sigma=1., gamma=1.):
        [n, _] = input.shape

        D = metrics.pairwise_distances(input)
        fact = 1 / (2 * sigma ** 2)
        K = np.exp(-fact*D**2)

        I = np.identity(n)
        N = np.ones((n,1))
        A = np.concatenate((np.concatenate((K + I * (1 / gamma), N), axis=1),
                           np.concatenate((N.transpose(), [[0]]), axis=1)),
                           axis = 0)
        y = np.concatenate((targets, [0]), axis = 0)

        x = np.linalg.solve(A,y)
        alpha = x[0:-1]
        beta = x[-1]

        reg = .5 / n * alpha @ K @ alpha.transpose()
        # reg = .5 * alpha @ K @ alpha.transpose()
        yhat = alpha @ K + np.repeat(beta, n)
        recon = .5 / n * np.sum((yhat - targets)**2)
        # recon = .5 * np.sum((yhat - targets)**2)
        loss = gamma * recon + reg

        return alpha, beta, loss, recon, reg


