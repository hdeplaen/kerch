import torch
import torch.nn as nn
import torch.nn.functional as F


class Hilbert(nn.Module):
    def __init__(self, size_in, num_kernels, size_out, sigma, kernel_type, sigma_trainable, points_trainable):
        super(Hilbert, self).__init__()
        self.size_in = size_in
        self.num_kernels = num_kernels
        self.all_sv = range(self.num_kernels)
        self.size_out = size_out
        self.sigma = sigma
        self.kernel_type = kernel_type
        self.sigma_trainable = sigma_trainable
        self.points_trainable = points_trainable

        self.model = nn.ModuleDict({
            # 'convkernel': ConvKernel(self.num_kernels),
            'expkernel': ExponentialKernel(self.size_in, self.num_kernels, self.sigma, self.sigma_trainable,
                                           self.points_trainable),
            'lin': CustomLinear(self.num_kernels, self.size_out)
        })

    def forward(self, x, idx_sv=None):
        if idx_sv is None:
            idx_sv = self.all_sv

        x = self.model[self.kernel_type](x, idx_sv)
        x = self.model['lin'](x, idx_sv)
        return x

    def _special_init(self, x, y):
        self.model[self.kernel_type]._special_init(x)
        self.model['lin']._special_init(y)

    def regularization_term(self, idx_sv=None):
        if idx_sv is None:
            idx_sv = self.all_sv

        alpha = self.model['lin'].weight[idx_sv]
        K = self.model[self.kernel_type].kernel_matrix(idx_sv)
        reg = (1 / len(idx_sv)) * alpha @ K @ alpha.t()
        return reg

class KPCA(nn.Module):
    def __init__(self):
        super(KPCA, self).__init__()

    def forward(self, x):
        return x

class CustomLinear(nn.Module):
    def __init__(self, size_in, size_out):
        super(CustomLinear, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.weight = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.size_in, self.size_out)), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # place on hyperplane (0-mean)
        weight = self.weight.data
        self.weight.data = weight - torch.mean(weight, dim=0).expand(self.size_in, -1)

    def forward(self, x, idx_sv):
        return x @ self.weight[idx_sv].t() + self.bias.expand(x.shape[0])

    def merge_kernels(self, idx1, idx2):
        self.size_in -= 1
        weight = self.weight.data
        weight[idx1] += weight[idx2]
        new_weight = torch.cat((weight[0:idx2], weight[idx2 + 1:]), dim=0)
        self.weight = nn.Parameter(new_weight, requires_grad=True)

    def suppress_kernel(self, idx_loc):
        self.size_in -= 1
        w = self.weight.data
        new_w = torch.cat((w[0:idx_loc], w[idx_loc + 1:]), dim=0)
        self.weight = nn.Parameter(new_w, requires_grad=True)

    def _special_init(self, y):
        w = nn.Parameter(y.clone()).detach()
        w -= torch.mean(w, dim=0).view(1, 1).expand(self.size_in, -1).squeeze()  # place on hyperplane
        self.weight.data = w


class ExponentialKernel(nn.Module):
    def __init__(self, size_in, num_kernels, sigma, sigma_trainable=False, points_trainable=True):
        # size_in: size of x
        # size_out: number of support vectors (alpha_i), number of neurons
        super(ExponentialKernel, self).__init__()
        self.size_in = size_in
        self.num_kernels = num_kernels
        self.sigma = sigma
        self.points_trainable = points_trainable

        # too close to center, take random points from different classes
        self.param = nn.Parameter(4 * nn.init.orthogonal_(torch.Tensor(self.size_in, self.num_kernels)),
                                  requires_grad=points_trainable)
        self.sigma_trainable = nn.Parameter(torch.tensor(self.sigma), requires_grad=sigma_trainable)

    def matrix(self, idx_sv):
        sv = self.param[:, idx_sv].t()
        K = self.forward(sv, idx_sv)
        return K

    def forward(self, x, idx_sv=None):
        num_points = x.size(0)
        xs = x[:, :, None].expand(-1, -1, len(idx_sv))
        params = self.param[:, idx_sv].expand(num_points, -1, -1)

        diff = xs - params
        norm2 = torch.sum(diff * diff, axis=1)
        fact = 1 / (2 * self.sigma_trainable ** 2)
        output = torch.exp(-fact * norm2)

        return output

    def merge_kernels(self, idx1, idx2):
        self.num_kernels -= 1
        p = self.param.data
        new_p = torch.cat((p[:, 0:idx2], p[:, idx2 + 1:]), dim=1)
        self.param = nn.Parameter(new_p, requires_grad=self.points_trainable)

    def suppress_kernel(self, idx_loc):
        self.num_kernels -= 1
        p = self.param.data
        new_p = torch.cat((p[:, 0:idx_loc], p[:, idx_loc + 1:]), dim=1)
        self.param = nn.Parameter(new_p, requires_grad=self.points_trainable)

    def _special_init(self, x):
        self.param.data = x.clone().t()


class ConvKernel(nn.Module):
    def __init__(self, num_kernels):
        super(ConvKernel, self).__init__()
        self.pixel_size = 8
        num_channels = 1
        capacity = 5
        num_latent = 7
        self.num_kernels = num_kernels

        self.conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=capacity, kernel_size=4, stride=1,
                                     padding=1)  # out: c x 64 x 64
        self.conv2 = torch.nn.Conv2d(in_channels=capacity, out_channels=capacity * 2, kernel_size=4, stride=2,
                                     padding=1)  # out: c x 32 x 32
        self.fc1 = torch.nn.Linear(90, num_latent)
        self.fc2 = torch.nn.Softmax()

        self.param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_kernels, self.pixel_size ** 2)) \
                                  .view(self.num_kernels, 1, self.pixel_size, self.pixel_size))

    def special_init(self, x):
        p = nn.Parameter(x.clone().view(self.num_kernels, 1, self.pixel_size, self.pixel_size))
        self.param.data = p

    def go_through(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc1(x)
        return x

    def forward(self, x):
        def cosine_distance_torch(x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

        x = x.view(x.shape[0], 1, self.pixel_size, self.pixel_size)
        x = self.go_through(x)
        sv = self.go_through(self.param)
        x = cosine_distance_torch(x, sv)
        return x
