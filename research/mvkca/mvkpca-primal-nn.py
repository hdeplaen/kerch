import kerch
import torch
from logging import INFO
from tqdm import trange
from matplotlib import pyplot as plt

# kerch.set_log_level(INFO)

DIM_FEATURES = 10
DIM_KPCA = 8
BATCH_SIZE = 800


########################################################################################################################
# NNs

class Encoder1(torch.nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.l0 = torch.nn.Linear(1, 25, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(25, 15, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(15, DIM_FEATURES, dtype=kerch.FTYPE)
        self.f = torch.nn.LeakyReLU()
        # self.f = torch.nn.Identity()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.l2(x)
        return x


class Decoder1(torch.nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.l0 = torch.nn.Linear(DIM_FEATURES, 15, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(15, 25, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(25, 1, dtype=kerch.FTYPE)
        self.f = torch.nn.LeakyReLU()
        # self.f = torch.nn.Identity()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.l2(x)
        return x


class Encoder2(torch.nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.l0 = torch.nn.Linear(1, 25, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(25, 15, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(15, DIM_FEATURES, dtype=kerch.FTYPE)
        self.f = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.f(self.l2(x))
        return x


class Decoder2(torch.nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.l0 = torch.nn.Linear(DIM_FEATURES, 15, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(15, 25, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(25, 1, dtype=kerch.FTYPE)
        self.f = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.f(self.l2(x))
        return x


mse = torch.nn.MSELoss()


########################################################################################################################
# TRAINING LOOP
# DATA
def _gen_data():
    t = 8 * torch.rand(BATCH_SIZE, dtype=kerch.FTYPE) - 4
    x = torch.sin(t)
    return x, t/4


# INIT
x, t = _gen_data()
x = x[:,None]
t = t[:,None]


# TRY SPACE AE
encx, decx = Encoder1(), Decoder1()
opt_encx = torch.optim.SGD(encx.parameters(), lr=5.e-3, weight_decay=1.e-2)
opt_decx = torch.optim.SGD(decx.parameters(), lr=5.e-3, weight_decay=1.e-2)
bar = trange(1500)
for epoch in bar:
    opt_encx.zero_grad()
    opt_decx.zero_grad()
    x_pred = decx.forward(encx.forward(x))
    l = mse(x, x_pred)
    l.backward()
    opt_encx.step()
    opt_decx.step()
    bar.set_description(f"{l.detach().cpu().numpy():1.2e}")
idx = torch.argsort(x.squeeze())
x = x[idx,:]
x_pred = x_pred[idx,:]
plt.plot(x.detach(), x_pred.detach())
plt.show()


## MV-KPCA
mdl = kerch.rkm.MVKPCA({"name": "space", "name": "explicit_nn", "_center": True, "network": encx},
                       {"name": "time", "name": "linear", "_center": True},
                       dim_output=DIM_KPCA, representation='primal',
                       param_trainable=False)
mdl.view('space').init_sample(x)
mdl.view('time').init_sample(t)
mdl.solve()
opt = kerch.opt.Optimizer(mdl)

# LOOP
# forward-backward
def _iter(x, t):
    # mdl.view('space').update_sample(x)
    # mdl.view('time').update_sample(t)
    loss1 = mdl.classic_loss()
    loss2 = mse(x[:,None], decx.forward(mdl.predict_sample('space')))
    loss3 = torch.tensor(0.) #mse(t, mdl.predict_sample('time'))
    # loss = loss1 + 100 * (loss2 + loss3)
    loss = loss2
    loss.backward(retain_graph=True)
    mdl.solve()
    return loss, loss1, loss2, loss3


# loop
bar = trange(30000)
for iter in bar:
    x, t = _gen_data()
    opt.zero_grad()
    loss, a, b, c = _iter(x, t)
    opt.step()
    if iter % 50 == 0:
        bar.set_description(f"loss: {loss.detach().cpu():1.2e},"
                            f"kpca: {a.detach().cpu():1.2e},"
                            f"ae-space: {b.detach().cpu():1.2e},"
                            f"ae-time: {c.detach().cpu():1.2e}")

########################################################################################################################
# RESULTS

x, t = _gen_data()  # test
t, idx = torch.sort(t)
x = x[idx]
x_pred = decx.forward(mdl.predict_oos({'time': t})).detach()

plt.plot(t, x_pred, label='prediction')
plt.plot(t, x, label='real')
plt.show()
