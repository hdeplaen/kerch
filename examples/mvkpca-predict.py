import numpy as np
import kerch
from logging import DEBUG, INFO, WARNING
import torch
from matplotlib import pyplot as plt

# PRELIMINARIES --------------------------------------------------------------------------------------------
kerch.set_log_level(WARNING)
torch.random.manual_seed(42)
NUM_POINTS = 5000
NUM_WEIGHTS = 150
DIM_OUTPUT = 10000
SPLIT_RATIO = 0.8

# DATA -----------------------------------------------------------------------------------------------------
x = torch.tensor(np.linspace(-10, 10, NUM_POINTS)).unsqueeze(1)
y = torch.tensor(np.sinc(x))

rand_idx = torch.randperm(x.shape[0])
train_len = int(len(rand_idx) * (1 - SPLIT_RATIO))

# train
train_x = x[rand_idx[:train_len], :]
train_y = y[rand_idx[:train_len], :]

# test
test_x = x[rand_idx[train_len:], :]
test_y = y[rand_idx[train_len:], :]

# MODEL
mdl = kerch.rkm.multiview.MVKPCA({"name": "space", "type": "random_features", "num_weights": NUM_POINTS, "center": False, "sample": train_y},
                       {"name": "time", "type": "random_features", "num_weights": NUM_POINTS, "center": False, "normalize": False, "sample": train_x},
                       center=False, dim_output=DIM_OUTPUT)
mdl.solve(representation='primal')


train_phi_yp = mdl.predict_oos({"time": train_x}).detach()
train_yp = mdl.view("space").kernel.phi_pinv(train_phi_yp)

test_phi_yp = mdl.predict_oos({"time": test_x}).detach()
test_yp = mdl.view("space").kernel.phi_pinv(test_phi_yp)

# plot ------------------------------------------------------
phi_x, idx = torch.sort(train_x, dim=0)
phi_y = train_y[idx, :]

phi_x_tilde_train, idx = torch.sort(train_x, dim=0)
phi_y_tilde_train = train_yp[idx, :]

phi_x_tilde_test, idx = torch.sort(test_x, dim=0)
phi_y_tilde_test = test_yp[idx, :]

plt.figure()
plt.plot(x.squeeze(), y.squeeze(), 'k--', label='Original function', lw=1)
plt.plot(phi_x.squeeze(), phi_y.squeeze(), 'kx', label='Train set', lw=1)
plt.plot(phi_x_tilde_train.squeeze(), phi_y_tilde_train.squeeze(), 'b+',
         label='Pred on train_set', lw=1)
plt.plot(phi_x_tilde_test.squeeze(), phi_y_tilde_test.squeeze(), 'r*',
         label='Pred on test_set', lw=1)
plt.title('RKM primal')
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
