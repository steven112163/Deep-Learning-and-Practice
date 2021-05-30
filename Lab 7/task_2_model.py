from torch.utils.checkpoint import checkpoint
import torch.nn.functional as func
import torch.distributions as dist
import torch.nn as nn
import torch
import math


class Actnorm(nn.Module):
    def __init__(self, param_dim=(1, 3, 1, 1)):
        super(Actnorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        return x * self.scale + self.bias, self.scale.abs().log().sum() * x.shape[2] * x.shape[3]

    def inverse(self, z):
        if not self.initialized:
            # Per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(z.transpose(0, 1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(z.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        x = (z - self.bias) / self.scale
        log_det = - self.scale.abs().log().sum() * z.shape[2] * z.shape[3]
        return x, log_det


class Invertible1x1Conv(nn.Module):
    def __init__(self, n_channels=3, lu_factorize=False):
        super(Invertible1x1Conv, self).__init__()
        self.lu_factorize = lu_factorize

        # Initialize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]

        if lu_factorize:
            # Compute LU factorization
            p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())

            # Initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('sign_s', s.sign())
            # store mask to compute LU in forward/inverse pass
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            log_det = self.log_s.sum() * H * W
        else:
            log_det = torch.slogdet(self.w)[-1] * H * W

        return func.conv2d(x, self.w.view(C, C, 1, 1)), log_det

    def inverse(self, z):
        B, C, H, W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            log_det = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            log_det = - torch.slogdet(self.w)[-1] * H * W

        return func.conv2d(z, w_inv.view(C, C, 1, 1)), log_det


class AffineCoupling(nn.Module):
    def __init__(self, n_channels, width):
        super(AffineCoupling, self).__init__()
        # network layers; per realnvp, network splits input, operates on half of it,
        # and returns shift and scale of dim = half the input channels
        # input is split along channel dim
        self.conv1 = nn.Conv2d(n_channels // 2, width, kernel_size=3, padding=1, bias=False)
        self.act_norm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.act_norm2 = Actnorm(param_dim=(1, width, 1, 1))
        # output is split into scale and shift components
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)
        # learned scale (cf RealNVP sec 4.1 / Glow official code
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels, 1, 1))

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x):
        # split along channel dim
        x_a, x_b = x.chunk(2, 1)

        h = func.relu(self.act_norm1(self.conv1(x_b))[0])
        h = func.relu(self.act_norm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        # shift; take even channels
        t = h[:, 0::2, :, :]
        # scale; take odd channels
        s = h[:, 1::2, :, :]
        # at initialization, s is 0 and sigmoid(2) is near identity
        s = torch.sigmoid(s + 2.)

        z_a = s * x_a + t
        z_b = x_b
        # concat along channel dim
        z = torch.cat([z_a, z_b], dim=1)

        log_det = s.log().sum([1, 2, 3])

        return z, log_det

    def inverse(self, z):
        # split along channel dim
        z_a, z_b = z.chunk(2, 1)

        h = func.relu(self.act_norm1(self.conv1(z_b))[0])
        h = func.relu(self.act_norm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        # shift; take even channels
        t = h[:, 0::2, :, :]
        # scale; take odd channels
        s = h[:, 1::2, :, :]
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        # concat along channel dim
        x = torch.cat([x_a, x_b], dim=1)

        log_det = - s.log().sum([1, 2, 3])

        return x, log_det


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims
        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussian while the other half undergo further transformations (cf RealNVP figure 4b).
    """

    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, log_det = self.gaussianize(x1, x2)
        return x1, z2, log_det

    def inverse(self, x1, z2):
        x2, log_det = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, log_det


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3,
                             padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(
            torch.zeros(2 * n_channels, 1, 1))  # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]  # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)  # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1, 2, 3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1, 2, 3])
        return x2, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account
        logdet = - math.log(256) * x[0].numel()
        return x - 0.5, logdet  # center x at 0

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.
        for module in self:
            x, logdet = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """

    def __init__(self, n_channels, width, lu_factorize=False):
        super().__init__(Actnorm(param_dim=(1, n_channels, 1, 1)),
                         Invertible1x1Conv(n_channels, lu_factorize),
                         AffineCoupling(n_channels, width))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """

    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4 * n_channels, width, lu_factorize) for _ in range(depth)],
                                        checkpoint_grads=checkpoint_grads)
        self.split = Split(4 * n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


# --------------------
# Model
# --------------------

class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""

    def __init__(self, width, depth, n_levels, input_dims=(4, 64, 64), checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4 ** (
                n_levels + 1) / 2 ** n_levels)  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / 2 ** (n_levels + 1))  # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList(
            [FlowLevel(in_channels * 2 ** i, width, depth, checkpoint_grads, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, width, lu_factorize) for _ in range(depth)],
                                       checkpoint_grads=checkpoint_grads)

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

        self.input_dims = input_dims
        self.label_to_condition = nn.Linear(24, input_dims[1] * input_dims[2], bias=True)

    def forward(self, x, label):
        condition = self.label_to_condition(label).view(-1, 1, self.input_dims[1], self.input_dims[2])
        x, sum_logdets = self.preprocess(x)
        x = torch.cat([x, condition], 1)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x)
            sum_logdets = sum_logdets + logdet
            zs.append(z)
        x = self.squeeze(x)
        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet

        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)
        return zs, sum_logdets

    def inverse(self, zs=None, batch_size=None, z_std=1.):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        # pass through inverse flow
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        x, logdet = self.flowstep.inverse(z)
        sum_logdets = sum_logdets + logdet
        x = self.squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs) == 1 else zs[
                -i - 2])  # if no z's are passed, generate new random numbers from the base dist
            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet
        # postprocess
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return dist.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, label, bits_per_pixel=False):
        zs, logdet = self.forward(x, label=label)
        log_prob = sum(self.base_dist.log_prob(z).sum([1, 2, 3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return log_prob
