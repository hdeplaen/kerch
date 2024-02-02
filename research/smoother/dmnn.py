# coding=utf-8
import kerch
import torch
from numpy import genfromtxt
from matplotlib import pyplot as plt

# SCORES
scores = genfromtxt('scores.csv', delimiter=";")
scores = torch.tensor(scores)
num_w, num_ex = scores.shape
weights = torch.linspace(0, 5, num_w)
smooth = torch.zeros_like(scores)

# SMOOTHER
sigma_min, sigma_max, sigma_step = 0.05, 1, 0.05
num_sigma = round((sigma_max - sigma_min) / sigma_step) + 1
sigmas = torch.linspace(sigma_min, sigma_max, num_sigma)

k = kerch.kernel.RBF(sample=weights)
for num in range(num_sigma):
    k.sigma = sigmas[num]
    for i in range(num_ex):
        smooth[:, i] = kerch.kernel.preimage.smoother(k.K, sample=scores[:,i]).squeeze()

    # PLOT
    fig = plt.figure(num)
    plt.plot(weights, smooth)
    plt.legend(['Exercise 1', 'Exercise 2', 'Exercise 3', 'Exercise 4'])
    plt.xlabel('Score')
    plt.ylabel('Student number')
    plt.title(f"Distributions (Sigma = {k.sigma:.2f})")
    fig.savefig(f"{num}.png")
    plt.show()
