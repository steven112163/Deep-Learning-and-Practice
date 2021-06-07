from typing import Tuple, Optional, List
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import torch


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
            bias = -torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True)
            v = torch.mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
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
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias

    def _scale(self, x: torch.Tensor, sld: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Scale the data
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Scaled data and sum of log-determinant
        """
        if not reverse:
            x *= torch.exp(self.logs)
        else:
            x *= torch.exp(-self.logs)

        if sld is not None:
            ld = self.logs.sum() * x.size(2) * x.size(3)
            if not reverse:
                sld += ld
            else:
                sld -= ld

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

        if not reverse:
            x = self._center(x=x, reverse=False)
            x, sld = self._scale(x=x, sld=sld, reverse=False)
        else:
            x, sld = self._scale(x=x, sld=sld, reverse=True)
            x = self._center(x=x, reverse=True)

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
        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]
        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones(num_channels, num_channels), -1)
        eye = torch.eye(num_channels, num_channels)

        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)
        self.l_mask = l_mask
        self.eye = eye

    def forward(self, x: torch.Tensor, sld: torch.tensor, reverse: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Invertible 1x1 convolution forwarding
        :param x: Batched data
        :param sld: Sum of log-determinant
        :param reverse: Reverse or not
        :return: Transformed data and sum of log-determinant
        """
        self.l_mask = self.l_mask.to(x.device)
        self.eye = self.eye.to(x.device)

        lower = self.lower * self.l_mask + self.eye

        u = self.upper * self.l_mask.transpose(0, 1).contiguous()
        u += torch.diag(self.sign_s * torch.exp(self.log_s))

        ld = torch.sum(self.log_s) * x.size(2) * x.size(3)

        if not reverse:
            weight = torch.matmul(self.p, torch.matmul(lower, u)).view(self.num_channels, self.num_channels, 1, 1)
            sld += ld
        else:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)

            weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv)).view(self.num_channels, self.num_channels, 1, 1)
            sld -= ld

        z = func.conv2d(input=x, weight=weight)

        return z, sld


def compute_same_pad(kernel_size: Optional[int or List[int]], stride: Optional[int or List[int]]) -> List[int]:
    """
    Compute paddings
    :param kernel_size: Kernel size
    :param stride: Stride
    :return: Paddings
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


class Conv2d(nn.Module):
    """
    Conv2d with actnorm
    :arg in_channels: Input channels
    :arg out_channels: Output channels
    :arg kernel_size: Kernel size
    :arg stride: Stride
    :arg padding: Padding
    :arg do_actnorm: Whether use actnorm
    :arg weight_std: Weight standard deviation
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Optional[int or Tuple[int, int]] = (3, 3),
            stride: Optional[int or Tuple[int, int]] = (1, 1),
            padding: str = "same",
            do_actnorm: bool = True,
            weight_std: float = 0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size=kernel_size, stride=stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm(in_channels=out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwarding
        :param x: Batched data
        :return: Batched data
        """
        output = self.conv(x)
        if self.do_actnorm:
            output, _ = self.actnorm.forward(x=output)
        return output


class Conv2dZeros(nn.Module):
    """
    Conv2d with zero initial weight and bias
    :arg in_channels: Input channels
    :arg out_channels: Output channels
    :arg kernel_size: Kernel size
    :arg stride: Stride
    :arg padding: Padding
    :arg logscale_factor: Log scale factor
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Optional[int or Tuple[int, int]] = (3, 3),
            stride: Optional[int or Tuple[int, int]] = (1, 1),
            padding: str = "same",
            logscale_factor: int = 3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size=kernel_size, stride=stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwarding
        :param x: Batched data
        :return: Batched data
        """
        output = self.conv(x)
        return output * torch.exp(self.logs * self.logscale_factor)


class LinearZeros(nn.Module):
    """
    Linear with zero initial weight and bias
    :arg in_channels: Input features
    :arg out_channels: Output features
    :arg logscale_factor: Log scale factor
    """

    def __init__(self, in_channels: int, out_channels: int, logscale_factor: int = 3):
        super().__init__()

        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwarding
        :param x: Batched data
        :return: Batched data
        """
        output = self.linear(x)
        return output * torch.exp(self.logs * self.logscale_factor)


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
        x_id, x_change = x[:, : x.size(1) // 2, ...], x[:, x.size(1) // 2:, ...]

        scale_and_translate = self.nn.forward(x=x_id, x_cond=x_cond)
        scale, translate = scale_and_translate[:, 0::2, ...], scale_and_translate[:, 1::2, ...]
        scale = torch.sigmoid(scale + 2.)

        # Scale and translate
        ld = torch.sum(scale.log(), dim=[1, 2, 3])
        if not reverse:
            x_change += translate
            x_change *= scale
            sld += ld
        else:
            x_change /= scale
            x_change -= translate
            sld -= ld

        x = torch.cat((x_id, x_change), dim=1)

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

        self.in_conv = Conv2d(in_channels=in_channels,
                              out_channels=mid_channels)
        self.in_cond_conv = Conv2d(in_channels=cond_channels,
                                   out_channels=mid_channels)

        self.mid_conv = Conv2d(in_channels=mid_channels,
                               out_channels=mid_channels,
                               kernel_size=(1, 1))
        self.mid_cond_conv = Conv2d(in_channels=mid_channels,
                                    out_channels=mid_channels,
                                    kernel_size=(1, 1))

        self.out_conv = Conv2dZeros(in_channels=mid_channels,
                                    out_channels=out_channels)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Compute scale and translate from batched data and condition
        :param x: Batched data
        :param x_cond: Batched condition
        :return: Scale and translate as one tensor
        """
        x_cond = self.in_cond_conv(x_cond)
        x = self.in_conv(x) + x_cond
        x = func.relu(x)

        x_cond = self.mid_cond_conv(x_cond)
        x = self.mid_conv(x) + x_cond
        x = func.relu(x)

        return self.out_conv(x)


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

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))

        self.flows = _CGlow(in_channels=3,
                            cond_channels=1,
                            mid_channels=num_channels,
                            num_levels=num_levels,
                            num_steps=num_steps)

        # Number of channels and image size after normal flow
        self.out_channels = 3 * (2 ** (num_levels - 1)) * 4
        self.out_image_size = image_size // (2 ** num_levels)

        # Project latent code to label
        self.project_latent = LinearZeros(in_channels=self.out_channels,
                                          out_channels=num_classes)

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

        mean = torch.zeros_like(z, device=z.device)
        logs = mean.clone()
        sld += GaussianDiag.log_prob(mean=mean, logs=logs, x=z)
        nll = (-sld) / float(np.log(2.0) * x.size(0) * x.size(1) * x.size(2))

        label_logits = self.project_latent(z.mean(2).mean(2)).view(-1, self.num_classes)
        label_logits = torch.sigmoid(label_logits)

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
                mean = torch.zeros(z_label.size(0),
                                   self.out_channels,
                                   self.out_image_size,
                                   self.out_image_size,
                                   device=z_label.device)
                logs = mean.clone()
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

    def _pre_process(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Preprocess x to x - 0.5 + U(0, 1/256)
        :param x: Batched data
        :return: Preprocessed data and sum of log-determinant
        """
        x += torch.zeros_like(x).uniform_(1.0 / 256)
        sld = float(-np.log(256.) * x.size(1) * x.size(2) * x.size(3)) * torch.ones(x.size(0), device=x.device)

        return x, sld


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

    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, nll: torch.Tensor, label_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        :param nll: Negative log-likelihood
        :param label_logits: Label logits
        :param labels: Labels
        :return: Loss
        """

        return func.binary_cross_entropy_with_logits(input=label_logits, target=labels.float()) * 0.5 + torch.mean(nll)
