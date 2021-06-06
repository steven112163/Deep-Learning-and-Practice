from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import torch


def mean_dim(tensor: torch.Tensor, dim: list = None, keepdims: bool = False) -> torch.Tensor:
    """
    Take the mean along multiple dimensions
    :param tensor: Tensor of values to average
    :param dim: List of dimensions along which to take the mean
    :param keepdims: Keep dimensions rather than squeezing
    :return: New tensor of mean value(s)
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


class ActNorm(nn.Module):
    """
    Activation normalization for 2D inputs.
    The bias and scale get initialized using the mean and variance of the first mini-batch.
    After the init, bias and scale are trainable parameters.
    Adapted from: https://github.com/openai/glow
    :arg in_channels: Number of channels in the input
    :arg scale: Scale factor for initial logs
    """

    def __init__(self, in_channels: int, scale: float = 1.):
        super(ActNorm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        self.num_features = in_channels
        self.scale = float(scale)
        self.eps = 1e-6
        self.is_initialized = False

    def initialize_parameters(self, x: torch.Tensor) -> None:
        """
        Initialize bias and logs
        :param x: First mini-batch
        :return: None
        """
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(tensor=x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim(tensor=(x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized = True

    def _center(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Translate the data
        :param x: Batched data
        :param reverse: Reverse or not
        :return: Translated data
        """
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x: torch.Tensor, sld: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Scale the data
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Scaled data and sum of log-determinant
        """
        if reverse:
            x = x * self.logs.mul(-1).exp()
        else:
            x = x * self.logs.exp()

        if sld is not None:
            ld = self.logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sld -= ld
            else:
                sld += ld

        return x, sld

    def forward(self, x: torch.Tensor, sld: torch.Tensor = None, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Actnorm forwarding
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Scaled and translated data & sum of log-determinant
        """
        if not self.is_initialized:
            self.initialize_parameters(x=x)

        if reverse:
            x = self._center(x=x, reverse=reverse)
            x, sld = self._scale(x=x, sld=sld, reverse=reverse)
        else:
            x, sld = self._scale(x=x, sld=sld, reverse=reverse)
            x = self._center(x=x, reverse=reverse)

        return x, sld


class InvConv(nn.Module):
    """
    Invertible 1x1 Convolution for 2D inputs.
    Originally described in Glow (https://arxiv.org/abs/1807.03039).
    Does not support LU-decomposed version.
    :arg num_channels: Number of channels in the input and output
    """

    def __init__(self, num_channels: int):
        super(InvConv, self).__init__()

        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x: torch.Tensor, sld: torch.tensor, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Invertible 1x1 convolution forwarding
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Transformed data and sum of log-determinant
        """
        ld = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sld -= ld
        else:
            weight = self.weight
            sld += ld

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = func.conv2d(input=x, weight=weight)

        return z, sld


class Coupling(nn.Module):
    """
    Affine coupling layer originally used in Real NVP and described by Glow.
    :arg in_channels: Number of channels in the input
    :arg cond_channels: Number of channels in the condition
    :arg mid_channels: Number of channels in the intermediate activation in NN
    """

    def __init__(self, in_channels: int, cond_channels: int, mid_channels: int):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels=in_channels,
                     cond_channels=cond_channels,
                     mid_channels=mid_channels,
                     out_channels=2 * in_channels)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, sld: torch.Tensor, reverse: bool = False) -> Tuple[
        torch.Tensor, ...]:
        """
        Affine coupling forwarding
        :param x: Batched data
        :param x_cond: Batched condition
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Affine coupled data and sum of log-determinant
        """
        x_change, x_id = x.chunk(2, dim=1)

        scale_and_translate = self.nn.forward(x=x_id, x_cond=x_cond)
        scale, translate = scale_and_translate[:, 0::2, ...], scale_and_translate[:, 1::2, ...]
        scale = torch.sigmoid(scale + 2.)

        # Scale and translate
        ld = torch.sum(scale.log(), dim=[1, 2, 3])
        if reverse:
            x_change = (x_change - translate) / scale
            sld -= ld
        else:
            x_change = x_change * scale + translate
            sld += ld

        x = torch.cat((x_change, x_id), dim=1)

        return x, sld


class NN(nn.Module):
    """
    Small convolutional network used to compute scale and translate factors.
    :arg in_channels: Number of channels in the input
    :arg cond_channels: Number of channels in the condition
    :arg mid_channels: Number of channels in the hidden activations
    :arg out_channels: Number of channels in the output
    """

    def __init__(self, in_channels: int, cond_channels: int, mid_channels: int, out_channels: int):
        super(NN, self).__init__()

        self.in_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=mid_channels,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
        self.in_cond_conv = nn.Conv2d(in_channels=cond_channels,
                                      out_channels=mid_channels,
                                      kernel_size=3,
                                      padding=1,
                                      bias=False)
        nn.init.normal_(self.in_conv.weight, mean=0., std=0.05)
        nn.init.normal_(self.in_cond_conv.weight, mean=0., std=0.05)
        self.in_norm = ActNorm(in_channels=mid_channels)

        self.mid_conv = nn.Conv2d(in_channels=mid_channels,
                                  out_channels=mid_channels,
                                  kernel_size=1,
                                  padding=0,
                                  bias=False)
        self.mid_cond_conv = nn.Conv2d(in_channels=cond_channels,
                                       out_channels=mid_channels,
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)
        nn.init.normal_(self.mid_conv.weight, mean=0., std=0.05)
        nn.init.normal_(self.mid_cond_conv.weight, mean=0., std=0.05)
        self.mid_norm = ActNorm(in_channels=mid_channels)

        self.out_conv = nn.Conv2d(in_channels=mid_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.register_parameter('logs', nn.Parameter(torch.zeros(out_channels, 1, 1)))

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Compute scale and translate from batched data and condition
        :param x: Batched data
        :param x_cond: Batched condition
        :return: Scale and translate as one tensor
        """
        x = self.in_conv(x) + self.in_cond_conv(x_cond)
        x, _ = self.in_norm(x)
        x = func.relu(x)

        x = self.mid_conv(x) + self.mid_cond_conv(x_cond)
        x, _ = self.mid_norm(x)
        x = func.relu(x)

        x = self.out_conv(x)

        return x * torch.exp(self.logs * 3)


class FlowStep(nn.Module):
    """
    Single flow step
    Forward: ActNorm -> InvConv -> Coupling
    Reverse: Coupling -> InvConv -> ActNorm
    :arg in_channels: Number of channels in the input
    :arg cond_channels: Number of channels in the condition
    :arg mid_channels: Number of hidden channels in the coupling layer
    """

    def __init__(self, in_channels: int, cond_channels: int, mid_channels: int):
        super(FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels=in_channels)
        self.conv = InvConv(num_channels=in_channels)
        self.coup = Coupling(in_channels=in_channels // 2,
                             cond_channels=cond_channels,
                             mid_channels=mid_channels)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, sld: torch.Tensor = None, reverse: bool = False) -> Tuple[
        torch.Tensor, ...]:
        """
        Single flow step
        Forward: ActNorm -> InvConv -> Coupling
        Reverse: Coupling -> InvConv -> ActNorm
        :param x: Batched data
        :param x_cond: Batched condition
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Batched data and sum of log-determinant
        """
        if not reverse:
            # Normal flow
            x, sld = self.norm.forward(x=x, sld=sld, reverse=False)
            x, sld = self.conv.forward(x=x, sld=sld, reverse=False)
            x, sld = self.coup.forward(x=x, x_cond=x_cond, sld=sld, reverse=False)
        else:
            # Reverse flow
            with torch.no_grad():
                x, sld = self.coup.forward(x=x, x_cond=x_cond, sld=sld, reverse=True)
                x, sld = self.conv.forward(x=x, sld=sld, reverse=True)
                x, sld = self.norm.forward(x=x, sld=sld, reverse=True)

        return x, sld


class CGlow(nn.Module):
    """
    Conditional Glow model
    :arg num_channels: Number of channels in the hidden layers
    :arg num_levels: Number of levels in the model (number of _CGlow class)
    :arg num_steps: Number of flow steps in each level
    :arg num_classes: Number of classes in the condition
    :arg image_size: Image size
    """

    def __init__(self, num_channels: int, num_levels: int, num_steps: int, num_classes: int, image_size: int):
        super(CGlow, self).__init__()

        self.flows = _CGlow(in_channels=3,
                            cond_channels=1,
                            mid_channels=num_channels,
                            num_levels=num_levels,
                            num_steps=num_steps)

        # Number of channels and image size after normal flow
        self.out_channels = 3 * (2 ** (num_levels - 1)) * 4
        self.out_image_size = image_size // (2 ** num_levels)

        # Project label to mean and variance
        self.project_label = nn.Linear(in_features=num_classes,
                                       out_features=self.out_channels * 2 * self.out_image_size ** 2)
        nn.init.zeros_(self.project_label.weight)
        nn.init.zeros_(self.project_label.bias)
        self.register_parameter('label_logs', nn.Parameter(torch.zeros(1, self.out_channels * 2, 1, 1)))

        # Project latent code to label
        self.project_latent = nn.Linear(in_features=self.out_channels,
                                        out_features=num_classes)
        nn.init.zeros_(self.project_label.weight)
        nn.init.zeros_(self.project_label.bias)
        self.register_parameter('latent_logs', nn.Parameter(torch.zeros(num_classes)))

        self.num_classes = num_classes
        self.image_size = image_size
        self.label_to_condition = nn.Sequential(
            nn.Linear(in_features=num_classes,
                      out_features=16 * 16,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16 * 16,
                      out_features=32 * 32,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32 * 32,
                      out_features=image_size * image_size)
        )

    def forward(self, x_label: torch.Tensor, x: torch.Tensor = None, reverse: bool = False) -> Optional[
        Tuple[torch.Tensor, ...] or torch.Tensor]:
        """
        CGlow forwarding
        :param x: Batched data
        :param x_label: Batched label
        :param reverse: Reverse or not
        :return: Batched data and sum of log-determinant
        """
        x_cond = self.label_to_condition(x_label).view(-1, 1, self.image_size, self.image_size)
        if not reverse:
            return self.normal_flow(x=x, x_label=x_label, x_cond=x_cond)
        else:
            return self.reverse_flow(z=None, z_label=x_label, z_cond=x_cond)

    def normal_flow(self, x: torch.Tensor, x_label: torch.Tensor, x_cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Normal flow
        :param x: Batched data
        :param x_label: Batched labels
        :param x_cond: Batched conditions
        :return: Batched latent & negative log likelihood & label logits
        """
        x, sld = self._pre_process(x)

        z, _, sld = self.flows.forward(x=x, x_cond=x_cond, sld=sld, reverse=False)

        mean, logs = self.get_mean_and_logs(label=x_label)
        sld += GaussianDiag.log_prob(mean=mean, logs=logs, x=z)
        nll = sld

        label_logits = self.project_latent(z.mean(2).mean(2)).view(-1, self.num_classes)
        label_logits *= torch.exp(self.latent_logs * 3)

        return z, nll, label_logits

    def reverse_flow(self, z_label: torch.Tensor, z_cond: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Reverse flow
        :param z: Latent code
        :param z_label: Batched labels
        :param z_cond: Batched conditions
        :return: Batched images
        """
        with torch.no_grad():
            if z is None:
                mean, logs = self.get_mean_and_logs(label=z_label)
                z = GaussianDiag.sample(mean, logs)
            sld = torch.zeros(z.size(0), device=z.device)

            # Get latent condition
            next_flow = self.flows
            while next_flow is not None:
                try:
                    z_cond = next_flow.squeeze_cond.forward(x=z_cond, reverse=False)
                    z_cond, _ = next_flow.split_cond.forward(x=z_cond, sld=sld, reverse=False)
                except AttributeError:
                    pass
                next_flow = next_flow.next

            x, _, _ = self.flows.forward(x=z, x_cond=z_cond, sld=sld, reverse=True)
        return x

    @staticmethod
    def _pre_process(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Preprocess x to x + U(0, 1/256)
        :param x: Batched data
        :return: Preprocessed data and sum of log-determinant
        """
        x += torch.zeros_like(x).uniform_(1.0 / 256)
        sld = float(-np.log(256.) * x.size(1) * x.size(2) * x.size(3)) * torch.ones(x.size(0), device=x.device)

        return x, sld

    def get_mean_and_logs(self, label: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Get mean and logs from label
        :param label: Batched label
        :return: Mean and logs
        """
        mean_and_logs = self.project_label(label).view(-1,
                                                       self.out_channels * 2,
                                                       self.out_image_size,
                                                       self.out_image_size)
        mean_and_logs *= torch.exp(self.label_logs * 3)
        return mean_and_logs[:, :self.out_channels, ...], mean_and_logs[:, self.out_channels:, ...]


class _CGlow(nn.Module):
    """
    Recursive constructor for a cGlow model.
    Each call creates a single level.
    :arg in_channels: Number of channels in the input
    :arg mid_channels: Number of channels in hidden layers of each step
    :arg num_levels: Number of levels to construct. Counter for recursion
    :arg num_steps: Number of steps of flow for each level
    """

    def __init__(self, in_channels: int, cond_channels: int, mid_channels: int, num_levels: int, num_steps: int):
        super(_CGlow, self).__init__()

        self.squeeze = Squeeze2d()
        self.squeeze_cond = Squeeze2d()
        self.steps = nn.ModuleList([FlowStep(in_channels=in_channels * 4,
                                             cond_channels=cond_channels * 4,
                                             mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        self.level = num_levels

        if num_levels > 1:
            self.split = Split2d(num_channels=in_channels * 4)
            self.split_cond = Split2d(num_channels=cond_channels * 4)
            self.next = _CGlow(in_channels=in_channels * 2,
                               cond_channels=cond_channels * 2,
                               mid_channels=mid_channels,
                               num_levels=num_levels - 1,
                               num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, sld: torch.Tensor, reverse: bool = False) -> Tuple[
        torch.Tensor, ...]:
        """
        Forwarding of each level
        :param x: Batched data
        :param x_cond: Batched condition
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Batched data & batched conditions & sum of log-determinant
        """
        if not reverse:
            return self.normal_flow(x=x, x_cond=x_cond, sld=sld)
        else:
            return self.reverse_flow(x=x, x_cond=x_cond, sld=sld)

    def normal_flow(self, x: torch.Tensor, x_cond: torch.Tensor, sld: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Normal flow
        :param x: Batched data
        :param x_cond: Batched condition
        :param sld: Sum of log-determinant
        :return: Batched data & batched conditions & sum of log-determinant
        """
        x = self.squeeze.forward(x=x, reverse=False)
        x_cond = self.squeeze_cond.forward(x=x_cond, reverse=False)

        for step in self.steps:
            x, sld = step.forward(x=x, x_cond=x_cond, sld=sld, reverse=False)

        try:
            x, sld = self.split.forward(x=x, sld=sld, reverse=False)
            x_cond, sld = self.split_cond.forward(x=x_cond, sld=sld, reverse=False)
            x, x_cond, sld = self.next.forward(x=x, x_cond=x_cond, sld=sld, reverse=False)
        except AttributeError:
            pass

        return x, x_cond, sld

    def reverse_flow(self, x: torch.Tensor, x_cond: torch.Tensor, sld: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Reverse flow
        :param x: Batched data
        :param x_cond: Batched condition
        :param sld: Sum of log-determinant
        :return: Batched data & batched conditions & sum of log-determinant
        """
        with torch.no_grad():
            try:
                x, x_cond, sld = self.next.forward(x=x, x_cond=x_cond, sld=sld, reverse=True)
                x, sld = self.split.forward(x=x, sld=sld, reverse=True)
                x_cond, sld = self.split_cond.forward(x=x_cond, sld=sld, reverse=True)
            except AttributeError:
                pass

            for step in reversed(self.steps):
                x, sld = step.forward(x=x, x_cond=x_cond, sld=sld, reverse=True)

            x = self.squeeze.forward(x=x, reverse=True)
            x_cond = self.squeeze_cond.forward(x=x_cond, reverse=True)

            return x, x_cond, sld


class Squeeze2d(nn.Module):
    """
    Trade spatial extent for channels.
    In forward direction, convert each  1x4x4 volume of input into a 4x1x1 volume of output.
    """

    def __init__(self):
        super(Squeeze2d, self).__init__()

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Squeeze forwarding
        :param x: Batched data
        :param reverse: Reverse or not
        :return: Squeezed/Un-squeezed data
        """
        batch_size, channel_size, height, width = x.size()
        if not reverse:
            # Squeeze
            x = x.view(batch_size, channel_size, height // 2, 2, width // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(batch_size, channel_size * 2 * 2, height // 2, width // 2)
        else:
            # Un-squeeze
            x = x.view(batch_size, channel_size // 4, 2, 2, height, width)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(batch_size, channel_size // 4, height * 2, width * 2)

        return x


class Split2d(nn.Module):
    """
    Split input into half channel
    :arg num_channels: Number of channels in the input
    """

    def __init__(self, num_channels: int):
        super(Split2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=num_channels // 2,
                              out_channels=num_channels,
                              kernel_size=3,
                              padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.register_parameter("logs", nn.Parameter(torch.zeros(num_channels, 1, 1)))

    def split2d_prior(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute mean and logs from data
        :param x: Batched data
        :return: mean and logs
        """
        mean_and_logs = self.conv(x)
        mean_and_logs = mean_and_logs * torch.exp(self.logs * 3)
        return mean_and_logs[:, 0::2, ...], mean_and_logs[:, 1::2, ...]

    def forward(self, x: torch.Tensor, sld: torch.Tensor = None, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Split forwarding
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Split/Un-split data and sum of log-determinant
        """
        if not reverse:
            x1, x2 = x[:, :x.size(1) // 2, ...], x[:, x.size(1) // 2:, ...]
            mean, logs = self.split2d_prior(x=x1)
            sld += GaussianDiag.log_prob(mean=mean, logs=logs, x=x2)
            return x1, sld
        else:
            x1 = x
            mean, logs = self.split2d_prior(x=x1)
            x2 = GaussianDiag.sample(mean=mean, logs=logs)
            x = torch.cat([x1, x2], dim=1)
            return x, sld


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean: torch.Tensor, logs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
        :param mean: Mean
        :param logs: Log std
        :param x: Batched data
        :return Log-likelihood
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def log_prob(mean: torch.Tensor, logs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Get log-likelihood for each batch
        :param mean: Mean
        :param logs: Log std
        :param x: Batched data
        :return: Batched log-likelihood
        """
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean: torch.Tensor, logs: torch.Tensor) -> torch.Tensor:
        """
        Sample data from Gaussian with mean and logs
        :param mean: Mean
        :param logs: Log std
        :return: Sampled data
        """
        return torch.normal(mean=mean, std=torch.exp(logs))


class NLLLoss(nn.Module):
    """
    Negative log-likelihood loss
    """

    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z: torch.Tensor, nll: torch.Tensor, label_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        :param z: latent code
        :param nll: Negative log-likelihood
        :param label_logits: Label logits
        :param labels: Labels
        :return: Loss
        """

        nll = nn.BCEWithLogitsLoss()(label_logits, labels.float()) * 0.5 + torch.mean(nll) - np.log(self.k) * np.prod(
            z.size()[1:])

        return -nll
