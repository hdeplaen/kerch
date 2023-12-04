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
import ray
from ray import tune, air
from ray.air.callbacks.wandb import WandbLoggerCallback

# PRELIMINARIES --------------------------------------------------------------------------------------------
# torch.random.manual_seed(42)


max_range, max_out_of_range = 10, 14
# f = lambda x: torch.sin((x+max_range+.1)**2)
f = torch.sin

kerch.set_log_level(ERROR)

METHOD = 'smoother'
REPRESENTATION = 'primal'
RECON_REPR = 'primal'
KNN = 1
NUM_POINTS = 500
NUM_WEIGHTS = 1000
DIM_OUTPUT = 500
SPLIT_RATIO = .8
DEV = torch.device('cpu')

# DATA -----------------------------------------------------------------------------------------------------
x = torch.tensor(np.linspace(-max_range, max_range, NUM_POINTS), device=DEV).unsqueeze(1)
y = f(x)

rand_idx = torch.randperm(x.shape[0])
train_len = int(len(rand_idx) * (SPLIT_RATIO))

# train
train_x = x[rand_idx[:train_len], :]
train_y = y[rand_idx[:train_len], :]
train_x, train_ind = torch.sort(train_x.squeeze())
train_y = train_y[train_ind,:]

# test
test_x = x[rand_idx[train_len:], :]
test_y = y[rand_idx[train_len:], :]
test_x, test_ind = torch.sort(test_x.squeeze())
test_y = test_y[test_ind,:]

# MODEL
sample_transforms = ["minmax_rescaling"]
kernel_transforms = []
mdl = kerch.rkm.multiview.MVKPCA(
    {"name": "space", "sample": train_y, "kappa": 1,
     "type": "rff", "base_type":"rbf", "sigma": 1,
     "kernel_transforms": kernel_transforms},
    {"name": "time", "sample": train_x, "kappa": 5,
     "type": "rff", "base_type": "rbf", "sigma": 5,
     "kernel_transforms": kernel_transforms},
                       center=False, dim_output=DIM_OUTPUT, representation=REPRESENTATION)
mdl.to(DEV)
mdl.solve()
# mdl.optimize(maxiter=100000)

out_of_range_x = np.concatenate((np.linspace(-max_out_of_range,-max_range,50), np.linspace(max_range,max_out_of_range,50)))

train_yp = mdl.predict({"time": train_x}, knn=KNN, representation=RECON_REPR, method=METHOD)['space'].detach()
test_yp = mdl.predict({"time": test_x}, knn=KNN, representation=RECON_REPR, method=METHOD)['space'].detach()
out_of_range = mdl.predict({"time": out_of_range_x}, representation=RECON_REPR, method=METHOD)['space'].detach()

MSE = torch.nn.MSELoss()
train_mse = MSE(train_y, train_yp)
test_mse = MSE(test_y, test_yp)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# plot ------------------------------------------------------
phi_x, idx = torch.sort(train_x, dim=0)
phi_y = train_y[idx, :]

phi_x_tilde_train, idx = torch.sort(train_x, dim=0)
phi_y_tilde_train = train_yp[idx, :]

phi_x_tilde_test, idx = torch.sort(test_x, dim=0)
phi_y_tilde_test = test_yp[idx, :]

plt.figure()
plt.plot(torch.linspace(-max_out_of_range,max_out_of_range,100), f(torch.linspace(-max_out_of_range,max_out_of_range,100)), 'k1--', label='Original function', lw=1)
plt.plot(phi_x.cpu().squeeze(), phi_y.cpu().squeeze(), 'kx', label='Train ground truth', lw=1)
plt.plot(phi_x_tilde_train.cpu().squeeze(), phi_y_tilde_train.cpu().squeeze(), 'b+',
         label='Train prediction', lw=1)
plt.plot(phi_x_tilde_test.cpu().squeeze(), phi_y_tilde_test.cpu().squeeze(), 'r*',
         label='Test prediction', lw=1)
plt.plot(out_of_range_x, out_of_range.cpu().squeeze(), 'g*',
         label='Out of range pediction', lw=1)
plt.title('RKM ' + REPRESENTATION)
plt.legend()
plt.grid()
plt.show()



# LEGACY ---------------------------------------------------------------------------------------------------

# # PLOT
# plt.figure()
# plt.plot(t,x_real)
# plt.plot(t,x_pred)
# plt.show()

# plt.figure()
# plt.bar(range(len(mdl.vals)), mdl.vals)
# plt.show()

# ws = mdl.view('space').weight.detach().numpy()
# plt.figure()
# plt.title('Space')
# for num in range(NUM_POINTS-1,NUM_POINTS):
#     plt.scatter(x_pred[0:num, 0], x_pred[0:num, 1], c='r')
#     plt.scatter(x_real[0:num, 0], x_real[0:num, 1], c='b')
#     plt.arrow(0, 0, ws[0, 0], ws[1, 0])
#     plt.arrow(0, 0, ws[0, 1], ws[1, 1])
#     plt.arrow(0, 0, ws[0, 2], ws[1, 2])
#     plt.show()
#
#
# wt = mdl.view('time').weight.detach().numpy()
# phi_t = mdl.view('time').Phi
#
# plt.figure()
# plt.title('Time')
# for num in range(NUM_POINTS-1,NUM_POINTS):
#     plt.scatter(phi_t[0:num, 0], phi_t[0:num, 1], c='r')
#     plt.arrow(0, 0, wt[0, 0], wt[1, 0])
#     plt.arrow(0, 0, wt[0, 1], wt[1, 1])
#     plt.arrow(0, 0, wt[0, 2], wt[1, 2])
#     plt.show()

# fig.show()
# plt.figure()
# plt.scatter(x_real[:, 2], x_real[:, 1], c='b')

# print(x_pred)
# print(x_real)

pass
