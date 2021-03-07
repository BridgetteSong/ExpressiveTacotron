# adapted from https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Networks.py
# MIT License

# Copyright (c) 2019 Jhosimar George Arias Figueroa

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F


# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Reshape layer
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        # https://zhuanlan.zhihu.com/p/50065712
        U = torch.rand(shape) # sample from uniform [0,1]
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


# Sample from a Gaussian distribution
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z

    # Inference Network


class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            nn.Linear(x_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            GumbelSoftmax(512, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            # print(concat.shape)
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        # x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        # p(z|y)
        y_mu, y_var = self.pzy(y)

        return (z, (z, mu, var, y_mu, y_var, prob, logits))


# GMVAE Network
class GMVAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, temperature, hard)

        return out_inf