from torch import nn
import torch.nn.functional as F

# pylint: disable=arguments-differ

### Blocks ###
class ConvBlock(nn.Sequential):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=False, norm="batch", actv="relu"):
        super(ConvBlock, self).__init__()
        # Convolution
        self.add_module("conv", nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding=padding, bias=bias))

        # Normalization
        norm_layer = get_norm_layer2d(output_nc, norm=norm)
        if norm_layer is not None:
            self.add_module("norm", norm_layer)

        # Activation
        actv_layer = get_activation_layer(actv)
        if actv_layer is not None:
            self.add_module("actv", actv_layer)

class DeConvBlock(nn.Sequential):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=False, norm="batch", actv="lrelu", method="deConv"):
        super(DeConvBlock, self).__init__()
        # Convolution
        if method == "convTrans":
            self.add_module("deconv", nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding=padding, bias=bias))
        elif method == "deConv":
            self.add_module("deconv", DeConvLayer(input_nc, output_nc))
        elif method == "pixlSuffle":
            raise NotImplementedError("PixelSuffle is not implemented")
        else:
            raise NameError("Unknown method: " + method)

        # Normalization
        norm_layer = get_norm_layer2d(output_nc, norm=norm)
        if norm_layer is not None:
            self.add_module("norm", norm_layer)

        # Activation
        actv_layer = get_activation_layer(actv)
        if actv_layer is not None:
            self.add_module("actv", actv_layer)



### Layers ###
class DeConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, bias=False):
        super(DeConvLayer, self).__init__()
        self.model = nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.model(x)

class FlattenLayer(nn.Module):
    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, -1)

class L2NormalizeLayer(nn.Module):
    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, p=2, dim=1)

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, *self.shape)

class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class GradientReverseLayer(nn.Module):
    def __init__(self, scale):
        super(GradientReverseLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x.clone()

    def backward(self, grad_out):
        return -self.scale * grad_out.clone()


# pylint: disable=abstract-method
### Loss ###
class KLLoss(nn.Module):
    def __call__(self, mu, logvar):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds.sum(1).mean(0)





def get_activation_layer(actv="relu"):
    layer = None
    if actv == "none":
        pass
    elif actv == "relu":
        layer = nn.ReLU()
    elif actv == "relu6":
        layer = nn.ReLU6()
    elif actv == "lrelu":
        layer = nn.LeakyReLU(0.2)
    elif actv == "tanh":
        layer = nn.Tanh()
    elif actv == "sigmoid":
        layer = nn.Sigmoid()
    else:
        raise NameError("Unknown activation: {}".format(actv))

    return layer

def get_norm_layer2d(nch, norm="batch"):
    layer = None
    if norm == "none":
        pass
    elif norm == "batch":
        layer = nn.BatchNorm2d(nch)
    elif norm == "instance":
        layer = nn.InstanceNorm2d(nch)
    else:
        raise NameError("Unknown normalization: {}".format(norm))

    return layer
