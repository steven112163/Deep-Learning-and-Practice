import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.
    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
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
    """Activation normalization for 2D inputs.
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """

    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
    Args:
        in_channels (int): Number of channels in the input and output.
        cond_channels (int):
    """

    def __init__(self, in_channels, cond_channels, mid_channels):
        super(InvConv, self).__init__()
        self.in_channels = in_channels
        self.nn = CondWNN(cond_channels, mid_channels, in_channels)

    def forward(self, x, x_cond, sldj, reverse=False):
        batch_size, channel_size, height, width = x.size()
        converter = nn.Linear(height * width, channel_size)

        # (batch_size, channel_size, height, width)
        weight = self.nn(x_cond)
        # (batch_size, channel_size, channel_size)
        weight = converter(weight.view(batch_size, channel_size, height * width))

        ldj = torch.slogdet(weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = weight
            sldj = sldj + ldj

        weight = weight.view(batch_size * channel_size, channel_size, 1, 1)
        x = x.view(1, batch_size * channel_size, height, width)
        z = F.conv2d(x, weight, groups=batch_size).view(batch_size, channel_size, height, width)

        return z, sldj


class CondWNN(nn.Module):
    """
    Small convolutional network used to compute weight matrix.
    Args:
        in_channels (int): Number of channels in the condition.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """

    def __init__(self, in_channels, mid_channels, out_channels, use_act_norm=False):
        super(CondWNN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_conv_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv_1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv_2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.in_conv(x)
        x = F.relu(x)

        x = self.mid_conv_1(x)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv_2(x)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """

    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = CondSTNN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = (x_change - t) * s.mul(-1).exp()
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = x_change * s.exp() + t
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class CondSTNN(nn.Module):
    """
    Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """

    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(CondSTNN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_cond_conv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_cond_conv.weight, 0., 0.05)

        self.mid_conv_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_cond_conv_1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv_1.weight, 0., 0.05)
        nn.init.normal_(self.mid_cond_conv_1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_cond_conv_2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv_2.weight, 0., 0.05)
        nn.init.normal_(self.mid_cond_conv_2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, x_cond):
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_cond_conv(x_cond)
        x = F.relu(x)

        x = self.mid_conv_1(x) + self.mid_cond_conv_1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv_2(x) + self.mid_cond_conv_2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x


class Glow(nn.Module):

    def __init__(self, num_channels, num_levels, num_steps, mode='sketch'):
        super(Glow, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.95], dtype=torch.float32))
        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           cond_channels=4,
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)
        self.mode = mode
        self.label_to_condition = nn.Linear(24, 64 * 64, bias=True)

    def forward(self, x, x_cond, reverse=False):
        x_cond = self.label_to_condition(x_cond).view(-1, 1, 64, 64)
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)
        if self.mode == 'gray':
            x_cond, _ = self._pre_process(x_cond)

        x = squeeze(x)
        x_cond = squeeze(x_cond)
        x, sldj = self.flows(x, x_cond, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        # y = x
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """

    def __init__(self, in_channels, cond_channels, mid_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              cond_channels=cond_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              cond_channels=4 * cond_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, x_cond, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, x_cond, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x_cond = squeeze(x_cond)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, x_cond, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
            x_cond = squeeze(x_cond, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, x_cond, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels, cond_channels, mid_channels)
        self.coup = Coupling(in_channels // 2, cond_channels, mid_channels)

    def forward(self, x, x_cond, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, x_cond, sldj, reverse)
            x, sldj = self.conv(x, x_cond, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, x_cond, sldj, reverse)
            x, sldj = self.coup(x, x_cond, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.
    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.
    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x


class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
                   - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
