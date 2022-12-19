import kerch
import numpy as np
from logging import DEBUG, INFO, WARNING, ERROR
import torch
from matplotlib import pyplot as plt

kerch.set_log_level(ERROR)

NUM_POINTS = 150
NUM_WEIGHTS = 9
DIM_OUTPUT = 20
SPLIT_RATIO = .8
DEV = torch.device('cpu')

# DATA -----------------------------------------------------------------------------------------------------
x = torch.tensor(np.linspace(-10, 10, NUM_POINTS), device=DEV).unsqueeze(1)
y = torch.tensor(torch.sin(x))

rand_idx = torch.randperm(x.shape[0])
train_len = int(len(rand_idx) * (1 - SPLIT_RATIO))

# train
train_x = x[rand_idx[:train_len], :]
train_y = y[rand_idx[:train_len], :]

# test
test_x = x[rand_idx[train_len:], :]
test_y = y[rand_idx[train_len:], :]

# oos
oos_x = torch.cat((torch.linspace(-15,-10,round(NUM_POINTS/2),),
                   torch.linspace(+10,+15,round(NUM_POINTS/2))))
oos_y = torch.sin(oos_x)

# MODEL
mdl = kerch.rkm.multiview.MVKPCA(
    {"name": "space", "name": "random_features", "num_weights": NUM_WEIGHTS, "_center": False, "sample": train_y},
    {"name": "time", "name": "random_features", "base_type": "rbf", "num_weights": NUM_WEIGHTS, "_center": False, "_normalize": False,
     "sample": train_x},
    center=False, dim_output=DIM_OUTPUT)
mdl.to(DEV)
mdl.solve(representation='primal')

train_phi_yp = mdl.predict_oos({"time": train_x}).detach()
train_yp = mdl.view("space").kernel.phi_pinv(train_phi_yp)

test_phi_yp = mdl.predict_oos({"time": test_x}).detach()
test_yp = mdl.view("space").kernel.phi_pinv(test_phi_yp)

oos_phi_yp = mdl.predict_oos({"time": oos_x}).detach()
oos_yp = mdl.view("space").kernel.phi_pinv(oos_phi_yp)

MSE = torch.nn.MSELoss()
train_mse = MSE(train_y, train_yp)
test_mse = MSE(test_y, test_yp)
oos_mse = MSE(oos_y, oos_yp)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"OOS MSE: {oos_mse}")

# plot ------------------------------------------------------
phi_x, idx = torch.sort(train_x, dim=0)
phi_y = train_y[idx, :]

phi_x_tilde_train, idx = torch.sort(train_x, dim=0)
phi_y_tilde_train = train_yp[idx, :]

phi_x_tilde_test, idx = torch.sort(test_x, dim=0)
phi_y_tilde_test = test_yp[idx, :]

plt.figure()
plt.plot(x.cpu().squeeze(), y.cpu().squeeze(), 'k--', label='Original function', lw=1)
plt.plot(oos_x[:round(NUM_POINTS/2)].cpu().squeeze(), oos_y[:round(NUM_POINTS/2)].cpu().squeeze(), 'k--', lw=1)
plt.plot(oos_x[round(NUM_POINTS/2):].cpu().squeeze(), oos_y[round(NUM_POINTS/2):].cpu().squeeze(), 'k--', lw=1)
plt.plot(phi_x.cpu().squeeze(), phi_y.cpu().squeeze(), 'kx', label='Train set', lw=1)
plt.plot(phi_x_tilde_train.cpu().squeeze(), phi_y_tilde_train.cpu().squeeze(), 'b+',
         label='Pred on train_set', lw=1)
plt.plot(phi_x_tilde_test.cpu().squeeze(), phi_y_tilde_test.cpu().squeeze(), 'r*',
         label='Pred on test_set', lw=1)
plt.plot(oos_x.cpu().squeeze(), oos_yp.cpu().squeeze(), 'g.',
         label='Pred on o-o-s', lw=1)
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
