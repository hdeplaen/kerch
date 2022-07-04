import kerch
import torch
from logging import INFO
from tqdm import trange

kerch.set_log_level(INFO)

DIM_FEATURES = 10
DIM_KPCA = 5
BATCH_SIZE = 100


########################################################################################################################
# NNs

class Encoder1(torch.nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.l0 = torch.nn.Linear(1, 25, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(25, 15, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(15, DIM_FEATURES, dtype=kerch.FTYPE)
        self.f = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.f(self.l2(x))
        return x


class Decoder1(torch.nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.l0 = torch.nn.Linear(DIM_FEATURES, 15, dtype=kerch.FTYPE)
        self.l1 = torch.nn.Linear(15, 25, dtype=kerch.FTYPE)
        self.l2 = torch.nn.Linear(25, 1, dtype=kerch.FTYPE)
        self.f = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.f(self.l0(x))
        x = self.f(self.l1(x))
        x = self.f(self.l2(x))
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
enc1, dec1 = Encoder1(), Decoder1()
enc2, dec2 = Encoder2(), Decoder2()

########################################################################################################################
# MV-KPCA

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "explicit_nn", "network": enc1},
                       {"name": "time", "type": "explicit_nn", "network": enc2},
                       dim_output=DIM_KPCA, representation='primal')


########################################################################################################################
# TRAINING LOOP
# DATA
def _gen_data():
    t = 8 * torch.rand(BATCH_SIZE, dtype=kerch.FTYPE) - 4
    x = torch.sin(t)
    return x, t


# INIT
x, t = _gen_data()
mdl.view('space').init_sample(x)
mdl.view('time').init_sample(t)
mdl.solve()
opt = kerch.opt.Optimizer(mdl)


# LOOP
# forward-backward
def _iter(x, t):
    mdl.view('space').update_sample(x)
    mdl.view('time').update_sample(t)
    loss1 = mdl.reconstruction_error()
    loss2 = mse(x, dec1.forward(mdl.reconstruct('space')))
    loss3 = mse(t, dec2.forward(mdl.reconstruct('time')))
    loss = loss1 + loss2 + loss3
    loss.backward()
    return loss


# loop
bar = trange(5000)
for iter in bar:
    x, t = _gen_data()
    loss = _iter(x, t)
    opt.step()
    bar.set_description(str(loss.detach().cpu()))
    if iter % 1000 == 0:
        pass

########################################################################################################################
# RESULTS

pass
