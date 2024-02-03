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

# SMOOTHER
sigma_min, sigma_max, sigma_step = 0.05, 1, 0.05
num_sigma = round((sigma_max - sigma_min) / sigma_step) + 1
sigmas = torch.linspace(sigma_min, sigma_max, num_sigma)


for num in range(num_sigma):
    smooth = kerch.method.kernel_smoother(domain=weights, observations=scores, sigma=sigmas[num])

    # PLOT
    fig = plt.figure(num)
    plt.plot(weights, smooth)
    plt.legend(['Exercise 1', 'Exercise 2', 'Exercise 3', 'Exercise 4'])
    plt.xlabel('Score')
    plt.ylabel('Student number')
    plt.title(f"Distributions (Sigma = {sigmas[num].item():.2f})")
    fig.savefig(f"{num}.png")
    plt.show()
