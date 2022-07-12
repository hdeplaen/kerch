import kerch
import torch
from torch.utils.data import Dataset, DataLoader
from logging import INFO
from tqdm import trange
from matplotlib import pyplot as plt

# kerch.set_log_level(INFO)

DIM_FEATURES = 10
DIM_KPCA = 8
BATCH_SIZE = 200
MAX_EPOCHS = 1000


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

mdl = kerch.rkm.MVKPCA({"name": "space", "type": "explicit_nn", "center": True, "network": enc1},
                       {"name": "time", "type": "linear", "center": True},
                       dim_output=DIM_KPCA, representation='primal')


########################################################################################################################
# Data

class SineDataset(Dataset):
    """Sine dataloader class"""

    def __init__(self, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.t = [i for i in torch.arange(0, 120, 0.1)]
        self.x = torch.tensor([torch.sin(i) for i in self.t])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]


sine_data = SineDataset()
sine_loader = DataLoader(sine_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         pin_memory=True,
                         num_workers=0)

# plot data for visualization
plt.figure()
plt.plot(sine_loader.dataset.x)
plt.show()
########################################################################################################################

# Initialize
mdl.view('space').init_sample(sine_loader.dataset.x)
mdl.view('time').init_sample(sine_loader.dataset.t)
mdl.solve()
opt = kerch.opt.Optimizer(mdl)

########################################################################################################################

# Training Loop
bar = trange(MAX_EPOCHS)
for iter in bar:

    for _, sample_batch in enumerate(sine_loader):  # Mini-batch

        opt.zero_grad()
        # loss, a, b = _iter(x=sample_batch[0], t=sample_batch[1])

        mdl.view('space').update_sample(sample_batch[0])
        mdl.view('time').update_sample(sample_batch[1])

        loss1 = mdl.classic_loss()
        loss2 = mse(sample_batch[0], dec1.forward(mdl.predict_sample('space')))
        # loss3 = mse(t, dec2.forward(mdl.predict_sample('time')))
        loss = loss1 + loss2  # + loss3

        loss.backward()
        opt.step()

    if iter % 50 == 0:
        bar.set_description(f"loss: {loss.detach().cpu():1.2e},"
                            f"kpca: {loss1.detach().cpu():1.2e},"
                            f"ae-space: {loss1.detach().cpu():1.2e}")

########################################################################################################################
# Results

x_pred = dec1.forward(mdl.reconstruct({'time': sine_loader.dataset.t})).detach()

plt.plot(sine_loader.dataset.t, x_pred, label='prediction')
plt.plot(sine_loader.dataset.t, sine_loader.dataset.x, label='real')
plt.show()
