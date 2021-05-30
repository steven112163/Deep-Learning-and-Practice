import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class ActNorm(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        size = [1, num_channels, 1, 1]

        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size) * 0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size) * 0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs), requires_grad=True))

    def forward(self, inputs, log_det=0, reverse=False):
        dimensions = inputs.size(2) * inputs.size(3)
        if not reverse:
            inputs = inputs + self.bias
            inputs = inputs * torch.exp(self.logs)
            d_log_det = torch.sum(self.logs) * dimensions
            log_det = log_det + d_log_det

        if reverse:
            inputs = inputs * torch.exp(-self.logs)
            inputs = inputs - self.bias
            d_log_det = - torch.sum(self.logs) * dimensions
            log_det = log_det + d_log_det

        return inputs, log_det


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1)):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                         padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.1)


class Conv2dResize(nn.Conv2d):

    def __init__(self, in_size, out_size):
        stride = [in_size[1] // out_size[1], in_size[2] // out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        super().__init__(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return [k0, k1]


class Conv2dNorm(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # initialize weight
        self.weight.data.normal_(mean=0.0, std=0.05)


class CondActNorm(nn.Module):

    def __init__(self, x_size, y_channels, x_hidden_channels, x_hidden_size):
        super().__init__()

        C_x, H_x, W_x = x_size

        # conditioning network
        self.x_Con = nn.Sequential(
            Conv2dResize(in_size=[C_x, H_x, W_x], out_size=[x_hidden_channels, H_x // 2, W_x // 2]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 2, W_x // 2],
                         out_size=[x_hidden_channels, H_x // 4, W_x // 4]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 4, W_x // 4],
                         out_size=[x_hidden_channels, H_x // 8, W_x // 8]),
            nn.ReLU()
        )

        self.x_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels * H_x * W_x // (8 * 8), x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, 2 * y_channels),
            nn.Tanh()
        )

    def forward(self, x, y, log_det=0, reverse=False):

        B, C, H, W = x.size()

        # generate weights
        x = self.x_Con(x)
        x = x.view(B, -1)
        x = self.x_Linear(x)
        x = x.view(B, -1, 1, 1)

        logs, bias = split_feature(x)
        dimensions = y.size(2) * y.size(3)

        if not reverse:
            # center and scale
            y = y + bias
            y = y * torch.exp(logs)
            d_log_det = dimensions * torch.sum(logs, dim=(1, 2, 3))
            log_det = log_det + d_log_det
        else:
            # scale and center
            y = y * torch.exp(-logs)
            y = y - bias
            d_log_det = - dimensions * torch.sum(logs, dim=(1, 2, 3))
            log_det = log_det + d_log_det

        return y, log_det


class Cond1x1Conv(nn.Module):

    def __init__(self, x_size, x_hidden_channels, x_hidden_size, y_channels):

        super().__init__()

        C_x, H_x, W_x = x_size

        # conditioning network
        self.x_Con = nn.Sequential(
            Conv2dResize(in_size=[C_x, H_x, W_x], out_size=[x_hidden_channels, H_x // 2, W_x // 2]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 2, W_x // 2],
                         out_size=[x_hidden_channels, H_x // 4, W_x // 4]),
            nn.ReLU(),
            Conv2dResize(in_size=[x_hidden_channels, H_x // 4, W_x // 4],
                         out_size=[x_hidden_channels, H_x // 8, W_x // 8]),
            nn.ReLU()
        )

        self.x_Linear = nn.Sequential(
            LinearZeros(x_hidden_channels * H_x * W_x // (8 * 8), x_hidden_size),
            nn.ReLU(),
            LinearZeros(x_hidden_size, x_hidden_size),
            nn.ReLU(),
            LinearNorm(x_hidden_size, y_channels * y_channels),
            nn.Tanh()
        )

    def get_weight(self, x, y, reverse):
        y_channels = y.size(1)
        B, C, H, W = x.size()

        x = self.x_Con(x)
        x = x.view(B, -1)
        x = self.x_Linear(x)
        weight = x.view(B, y_channels, y_channels)

        dimensions = y.size(2) * y.size(3)
        d_log_det = torch.slogdet(weight)[1] * dimensions

        if not reverse:
            weight = weight.view(B, y_channels, y_channels, 1, 1)

        else:
            weight = torch.inverse(weight.double()).float().view(B, y_channels, y_channels, 1, 1)

        return weight, d_log_det

    def forward(self, x, y, log_det=None, reverse=False):

        weight, d_log_det = self.get_weight(x, y, reverse)
        B, C, H, W = y.size()
        y = y.view(1, B * C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.view(B_k * C_i_k, C_o_k, H_k, W_k)

        if not reverse:
            z = F.conv2d(y, weight, groups=B)
            z = z.view(B, C, H, W)
            if log_det is not None:
                log_det = log_det + d_log_det

            return z, log_det
        else:
            z = F.conv2d(y, weight, groups=B)
            z = z.view(B, C, H, W)

            if log_det is not None:
                log_det = log_det - d_log_det

            return z, log_det


class Conv2dNormy(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1)):
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=False)

        # initialize weight
        self.weight.data.normal_(mean=0.0, std=0.05)
        self.act_norm = ActNorm(out_channels)

    def forward(self, inputs):
        x = super().forward(inputs)
        x, _ = self.act_norm(x)
        return x


class Conv2dZerosy(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1)):
        padding = [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2]
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.log_scale_factor = 3.0
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("new_bias", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        # init
        self.weight.data.zero_()
        self.new_bias.data.zero_()

    def forward(self, inputs):
        output = super().forward(inputs)
        output = output + self.new_bias
        output = output * torch.exp(self.logs * self.log_scale_factor)
        return output


class CondAffineCoupling(nn.Module):

    def __init__(self, x_size, y_size, hidden_channels):
        super().__init__()

        self.resize_x = nn.Sequential(
            Conv2dZeros(x_size[0], 16),
            nn.ReLU(),
            Conv2dResize((16, x_size[1], x_size[2]), out_size=y_size),
            nn.ReLU(),
            Conv2dZeros(y_size[0], y_size[0]),
            nn.ReLU()
        )

        self.f = nn.Sequential(
            Conv2dNormy(y_size[0] * 2, hidden_channels),
            nn.ReLU(),
            Conv2dNormy(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(),
            Conv2dZerosy(hidden_channels, 2 * y_size[0]),
            nn.Tanh()
        )

    def forward(self, x, y, log_det=0.0, reverse=False):

        z1, z2 = split_feature(y, "split")
        x = self.resize_x(x)

        h = torch.cat((x, z1), dim=1)
        h = self.f(h)
        shift, scale = split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2.)
        if not reverse:
            z2 = z2 + shift
            z2 = z2 * scale
            log_det = torch.sum(torch.log(scale), dim=(1, 2, 3)) + log_det

        if reverse:
            z2 = z2 / scale
            z2 = z2 - shift
            log_det = -torch.sum(torch.log(scale), dim=(1, 2, 3)) + log_det

        z = torch.cat((z1, z2), dim=1)

        return z, log_det


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, inputs, log_det=None, reverse=False):
        if not reverse:
            output = SqueezeLayer.squeeze2d(inputs, self.factor)
            return output, log_det
        else:
            output = SqueezeLayer.unsqueeze2d(inputs, self.factor)
            return output, log_det

    @staticmethod
    def squeeze2d(inputs, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return inputs
        B, C, H, W = inputs.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = inputs.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(inputs, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return inputs
        B, C, H, W = inputs.size()
        assert C % factor2 == 0, "{}".format(C)
        x = inputs.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // factor2, H * factor, W * factor)
        return x


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv2dZeros(num_channels // 2, num_channels),
            nn.Tanh()
        )

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, inputs, log_det=0., reverse=False, eps_std=None):
        if not reverse:
            z1, z2 = split_feature(inputs, "split")
            mean, logs = self.split2d_prior(z1)
            log_det = GaussianDiag.logp(mean, logs, z2) + log_det

            return z1, log_det
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = torch.cat([z1, z2], dim=1)

            return z, log_det


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=(1, 2, 3))

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsample(batch_size, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sample(mean, logs, eps_std)
        for i in range(1, batch_size):
            s = GaussianDiag.sample(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample


class LinearZeros(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, inputs):
        output = super().forward(inputs)
        return output


class LinearNorm(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.normal_(mean=0.0, std=0.1)
        self.bias.data.normal_(mean=0.0, std=0.1)


class CondGlowStep(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels):

        super().__init__()

        # 1. cond-actnorm
        self.act_norm = CondActNorm(x_size=x_size, y_channels=y_size[0], x_hidden_channels=x_hidden_channels,
                                    x_hidden_size=x_hidden_size)

        # 2. cond-1x1conv
        self.inv_conv = Cond1x1Conv(x_size=x_size, x_hidden_channels=x_hidden_channels,
                                    x_hidden_size=x_hidden_size, y_channels=y_size[0])

        # 3. cond-affine
        self.affine = CondAffineCoupling(x_size=x_size, y_size=[y_size[0] // 2, y_size[1], y_size[2]],
                                         hidden_channels=y_hidden_channels)

    def forward(self, x, y, log_det=None, reverse=False):

        if not reverse:
            # 1. cond-actnorm
            y, log_det = self.act_norm(x, y, log_det, reverse=False)

            # 2. cond-1x1conv
            y, log_det = self.inv_conv(x, y, log_det, reverse=False)

            # 3. cond-affine
            y, log_det = self.affine(x, y, log_det, reverse=False)

            # Return
            return y, log_det

        if reverse:
            # 3. cond-affine
            y, log_det = self.affine(x, y, log_det, reverse=True)

            # 2. cond-1x1conv
            y, log_det = self.inv_conv(x, y, log_det, reverse=True)

            # 1. cond-actnorm
            y, log_det = self.act_norm(x, y, log_det, reverse=True)

            # Return
            return y, log_det


class CondGlow(nn.Module):

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, K, L):

        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        C, H, W = y_size

        for l in range(0, L):

            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            y_size = [C, H, W]
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K CGlowStep
            for k in range(0, K):
                self.layers.append(CondGlowStep(x_size=x_size,
                                                y_size=y_size,
                                                x_hidden_channels=x_hidden_channels,
                                                x_hidden_size=x_hidden_size,
                                                y_hidden_channels=y_hidden_channels,
                                                )
                                   )

                self.output_shapes.append([-1, C, H, W])

            # 3. Split
            if l < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, x, y, log_det=0.0, reverse=False, eps_std=1.0):
        if not reverse:
            return self.encode(x, y, log_det)
        else:
            return self.decode(x, y, log_det, eps_std)

    def encode(self, x, y, log_det=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, Split2d) or isinstance(layer, SqueezeLayer):
                y, log_det = layer(y, log_det, reverse=False)

            else:
                y, log_det = layer(x, y, log_det, reverse=False)
        return y, log_det

    def decode(self, x, y, log_det=0.0, eps_std=1.0):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                y, log_det = layer(y, logdet=log_det, reverse=True, eps_std=eps_std)

            elif isinstance(layer, SqueezeLayer):
                y, log_det = layer(y, logdet=log_det, reverse=True)

            else:
                y, log_det = layer(x, y, logdet=log_det, reverse=True)

        return y, log_det


class CondGlowModel(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, x_size, y_size, x_hidden_channels, x_hidden_size, y_hidden_channels, flow_depth, num_levels,
                 learn_top, y_bins):
        super().__init__()
        self.flow = CondGlow(x_size=x_size,
                             y_size=y_size,
                             x_hidden_channels=x_hidden_channels,
                             x_hidden_size=x_hidden_size,
                             y_hidden_channels=y_hidden_channels,
                             K=flow_depth,
                             L=num_levels,
                             )

        self.learn_top = learn_top

        self.register_parameter("new_mean",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.register_parameter("new_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.n_bins = y_bins
        self.label_to_condition = nn.Linear(24, x_size[0] * x_size[1] * x_size[2], bias=True)

    def prior(self):

        if self.learn_top:
            return self.new_mean, self.new_logs
        else:
            return torch.zeros_like(self.new_mean), torch.zeros_like(self.new_mean)

    def forward(self, x=0.0, y=None, eps_std=1.0, reverse=False):
        y = self.label_to_condition(y).view(-1, 3, 64, 64)
        if not reverse:
            dimensions = y.size(1) * y.size(2) * y.size(3)
            log_det = torch.zeros_like(y[:, 0, 0, 0])
            log_det += float(-np.log(self.n_bins) * dimensions)
            z, objective = self.flow(x, y, log_det=log_det, reverse=False)
            mean, logs = self.prior()
            objective += GaussianDiag.logp(mean, logs, z)
            nll = -objective / float(np.log(2.) * dimensions)
            return z, nll

        else:
            with torch.no_grad():
                mean, logs = self.prior()
                if y is None:
                    y = GaussianDiag.batchsample(x.size(0), mean, logs, eps_std)
                y, log_det = self.flow(x, y, eps_std=eps_std, reverse=True)
            return y, log_det
