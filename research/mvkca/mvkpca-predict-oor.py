from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

import kerch
import numpy as np
from logging import DEBUG, INFO, WARNING, ERROR
import torch
from matplotlib import pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import os
import subprocess

# PRELIMINARIES --------------------------------------------------------------------------------------------
# torch.random.manual_seed(42)


max_range, max_out_of_range = 10, 14
# f = lambda x: torch.sin((x+max_range+.1)**2)
f = torch.sin
# f = lambda x : torch.sin(x * 2 * torch.pi)

kerch.set_log_level(ERROR)

REPR = 'dual'
RECON_REPR = 'dual'
SIGMA = .1
STEP = 1
NUM_OOR = int(500*.4)
NUM_POINTS = 500
NUM_WEIGHTS = 1000
DIM_OUTPUT = 1000
KNN = 100
DEV = torch.device('cpu')

# DATA -----------------------------------------------------------------------------------------------------
x = torch.tensor(np.linspace(-max_range, max_range, NUM_POINTS), device=DEV).unsqueeze(1)
y = f(x)

x_oor = torch.linspace(max_range, max_out_of_range, NUM_OOR).unsqueeze(1)
y_oor = f(x_oor)

# MODEL
sample_transforms = []
kernel_transforms = []

x_train = torch.concat((x, x), dim=0)
y_train = torch.concat((y, y), dim=0)

for idx in tqdm(range(0, NUM_OOR, STEP)):
    mdl = kerch.rkm.multiview.MVKPCA(
        {"name": "space", "kappa": 1., "type": "rbf", "sigma": .1,
         "sample": y_train,
         "sample_transforms": sample_transforms, "kernel_transforms": kernel_transforms},
        {"name": "time", "kappa": 1., "type": "rbf",
         "sigma": SIGMA, "sample": x_train,
         "sample_transforms": sample_transforms, "kernel_transforms": kernel_transforms},
        dim_output=DIM_OUTPUT, representation=REPR)
    mdl.to(DEV)
    mdl.solve()

    y_pred = mdl.predict({"time": x}, knn=KNN, representation=RECON_REPR)['space'].detach()
    y_pred_oor = mdl.predict({"time": x_oor}, knn=KNN, representation=RECON_REPR)['space'].detach()

    x_train = torch.concat((x, x, x_oor[:idx,:]), dim=0)
    y_train = torch.concat((y, y, y_pred_oor[:idx,:]), dim=0)

    ## PLOT

    plt.figure(0)
    plt.clf()

    # ground truths
    plt.plot(x.cpu().squeeze(), y.cpu().squeeze(), 'k1-', label='Train ground truth', lw=2)
    plt.plot(x_oor.cpu().squeeze(), y_oor.cpu().squeeze(), 'k1--', label='Out-of-range ground truth', lw=2)

    # predictions
    plt.plot(x.cpu().squeeze(), y_pred.cpu().squeeze(), 'b-', label="Train predictions", lw=3)
    plt.plot(x_oor[:idx,:].cpu().squeeze(), y_pred_oor[:idx,:].cpu().squeeze(), 'g-',
             label='OOR predictions used in update', lw=3)
    plt.plot(x_oor[idx:,:].cpu().squeeze(), y_pred_oor[idx:, :].cpu().squeeze(), 'r-',
             label='OOR predictions still to predict', lw=3)
    plt.title('Online RKM primal')
    plt.ylim([-2,2])
    plt.legend(loc='upper left')
    plt.grid()
    # plt.show()
    plt.savefig('img/tmp/' + str(idx) + '.png')

    if idx==0:
        subprocess.call(('xdg-open', 'img/tmp/0.png'))
        input("\nPress Enter to continue...")


gifname = 'img/mvkpca-predict-oor' \
          + '-STEP_' +  str(STEP) \
          + '-SIGMA_' + str(SIGMA) + '.gif'
with imageio.get_writer(gifname, mode="I") as writer:
    for idx in range(0, NUM_OOR, STEP):
        filename = 'img/tmp/' + str(idx) + '.png'
        image = imageio.imread(filename)
        writer.append_data(image)
        os.remove(filename)

subprocess.call(('xdg-open', gifname))