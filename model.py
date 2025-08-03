import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F
from model_utils import Bottleneck, HCE
import pywt
import pywt.data
from functools import partial

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(1, stride=1)
        self.Updim = Conv(ch_int, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        W_local = self.W_l(l)  # local feature from Local Feature Block
        W_global = self.W_g(g)  # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        g = self.sigmoid(max_out + avg_out) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# Window splitting and merging functions
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: torch.Tensor = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


### Global Feature Block

class Rearrange(nn.Module):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_lengths)


# If einops is not installed, this function needs to be added.
def rearrange(tensor, pattern, **axes_lengths):
    return tensor.permute(0, 2, 3, 1).reshape(*tensor.shape[:2], -1)


class Global_block(nn.Module):
    r""" Global Feature Block from modified Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=3, shift_size=0,  # 调整窗口大小为3×3，适应24×9的空间尺寸
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Modify the MLP to a standard structure
        mlp_hidden_dim = int(dim * mlp_ratio)  # Use the standard expansion ratio of 4 times
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: input features with shape (B, L, C)
            H, W: spatial dimensions (height, width) of the feature map
        """
        B, C, H, W = x.shape

        shortcut = x
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad the feature map so that its size is an integer multiple of the window size.
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        # print("x1: ", x.shape)
        # Circular shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # Split window
        # print("shifted_x: ", shifted_x.shape)
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        # print("x_windows: ", x_windows.shape)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # Calculate attention
        # print(x_windows.shape)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # Merge window
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # Reverse circular shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        B, H, W, C = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, -1, C)
        # print(x.shape)
        x = x.view(B, C, H, W)
        # Use the standard MLP structure
        # print(x.shape)
        # print(shortcut.shape)
        x = shortcut + self.drop_path(x)
        x_ = x.permute(0, 1, 3, 2).reshape(B, -1, C)
        x_ = self.drop_path(self.mlp(self.norm2(x_)))
        x_ = x_.view(B, C, H, W)
        x = x + x_

        return x


class Local_block(nn.Module):
    r""" Local Feature Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_rate=0.,groups=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)  # Depthwise separable convolution, processing spatial features
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)  # Pointwise convolution, implemented with linear layers for greater efficiency
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x


# _______________________________________________________________________________________________________________________
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)],
                               dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)],
                               dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.base_conv = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   padding='same',
                                   stride=1,
                                   dilation=1,
                                   groups=in_channels,
                                   bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4,
                       in_channels * 4,
                       kernel_size,
                       padding='same',
                       stride=1,
                       dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None
    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    def forward(self, x):
        return torch.mul(self.weight, x)


class EnhancedCrossAttentionHFF(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out, context_dim=None, num_heads=16, drop_rate=0., use_linear_attn=True):
        super().__init__()

        # Add handling of context feature dimensions
        self.context_dim = context_dim  # The number of channels of context features

        # Feature alignment
        self.local_align = nn.Sequential(
            nn.Conv2d(ch_1, ch_int, 1),
            LayerNorm(ch_int, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(ch_int, ch_int, 1)
        )

        self.global_align = nn.Sequential(
            nn.Conv2d(ch_2, ch_int, 1),
            LayerNorm(ch_int, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(ch_int, ch_int, 1)
        )

        # Multi - head attention parameters
        self.num_heads = num_heads
        self.head_dim = ch_int // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_linear_attn = use_linear_attn

        # Q, K, V projections
        self.q_proj = nn.Linear(ch_int, ch_int)
        self.k_proj = nn.Linear(ch_int, ch_int)
        self.v_proj = nn.Linear(ch_int, ch_int)

        # Output the projection
        self.out_proj = nn.Linear(ch_int, ch_int)

        # Retain the original attention mechanism
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_int, ch_int // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_int // 8, ch_int, 1, bias=False),
            nn.Sigmoid()
        )

        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            # nn.Conv2d(ch_int, ch_int, kernel_size=3, padding=1, groups=ch_int),
            WTConv2d(ch_int, ch_int, kernel_size=3, stride=1, bias=False, wt_levels=1, wt_type='db1'),
            nn.BatchNorm2d(ch_int),
            nn.GELU(),
            nn.Conv2d(ch_int, ch_int, kernel_size=1),
            nn.BatchNorm2d(ch_int)
        )

        # Context feature processing
        if context_dim is not None:
            self.context_transform = nn.Sequential(
                nn.Conv2d(context_dim, ch_int, kernel_size=1),
                nn.BatchNorm2d(ch_int),
                nn.GELU()
            )
        else:
            self.context_transform = None

        # Dynamic Fusion
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # Final fusion
        self.final_fusion = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.norm = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def linear_attention(self, q, k, v):
        """Linear complexity attention calculation"""
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        return out

    def forward(self, local_feat, global_feat, context_feat=None):
        B, C1, H, W = local_feat.shape

        # Feature alignment
        local_aligned = self.local_align(local_feat)
        global_aligned = self.global_align(global_feat)

        # Save the original features for residual connection
        local_orig = local_feat
        global_orig = global_feat

        # Calculate cross-attention
        # Flattened features
        local_flat = local_aligned.flatten(2).transpose(1, 2)  # [B, H*W, C]
        global_flat = global_aligned.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 投影
        q = self.q_proj(global_flat)
        k = self.k_proj(local_flat)
        v = self.v_proj(local_flat)

        # Multi - head Attention
        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention
        if self.use_linear_attn:
            attn_output = self.linear_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_output = (attn @ v)

        # Reshape
        cross_attn = attn_output.transpose(1, 2).reshape(B, H * W, -1)
        cross_attn = self.out_proj(cross_attn).transpose(1, 2).reshape(B, -1, H, W)

        # Original spatial attention
        max_feat, _ = torch.max(local_aligned, dim=1, keepdim=True)
        avg_feat = torch.mean(local_aligned, dim=1, keepdim=True)
        spatial_input = torch.cat([max_feat, avg_feat], dim=1)
        spatial_weight = self.spatial_attn(spatial_input)
        spatial_attn = local_aligned * spatial_weight

        # Original channel attention
        channel_weight = self.channel_attn(global_aligned)
        channel_attn = global_aligned * channel_weight

        # Dynamically fuse three types of attention
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        fused_attn = (
                fusion_weights[0] * cross_attn +
                fusion_weights[1] * spatial_attn +
                fusion_weights[2] * channel_attn
        )

        # Feature enhancement
        enhanced_feat = self.feature_enhancer(fused_attn) + fused_attn

        # Final fusion
        concat_feat = torch.cat([local_orig, global_orig, enhanced_feat], dim=1)

        # Normalization and final fusion
        concat_feat = self.norm(concat_feat)
        output = self.final_fusion(concat_feat)

        # If there are context features, first adapt the number of channels and then fuse them with it
        if context_feat is not None and self.context_transform is not None:
            # Upsample to the current resolution
            context_feat = F.interpolate(context_feat, size=(H, W))

            # Channel adaptation
            context_feat = self.context_transform(context_feat)

            # fusion
            output = output + context_feat

        # Residual connection
        output = enhanced_feat + self.drop_path(output)

        return output


# -----------------------------------------------------------------------------------------------------------------------
class Multi_FUE(nn.Module):
    def __init__(self, channel=512, dim=128,reduction=16, HFF_dp=0.,group_ratio=1,num_heads=16,window_size=3,shift_size=1,mlp_ratio=4):
        super(Multi_FUE, self).__init__()

        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        #self.FC13 = Local_block(dim=channel, groups=channel//group_ratio)
        self.FC13.apply(weights_init_kaiming)
        # self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        # self.FC12.apply(weights_init_kaiming)

        self.FC11 = Local_block(dim=channel,groups=channel)
        #self.FC11.apply(weights_init_kaiming)
        self.FC12 = Global_block(dim=channel, num_heads=num_heads, window_size=window_size, shift_size=shift_size, drop_path=0.1,mlp_ratio=mlp_ratio)
        #self.FC12.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        #self.FC23 = Local_block(dim=channel, groups=channel // group_ratio)
        self.FC23.apply(weights_init_kaiming)
        # self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        # self.FC12.apply(weights_init_kaiming)

        self.FC21 = Local_block(dim=channel,groups=channel)
        #self.FC21.apply(weights_init_kaiming)
        self.FC22 = Global_block(dim=channel, num_heads=num_heads, window_size=window_size, shift_size=shift_size, drop_path=0.1,mlp_ratio=mlp_ratio)
        #self.FC22.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)

        # iAFF modules for fusing x with x1 and then with x2
        self.fu1 = EnhancedCrossAttentionHFF(ch_1=dim, ch_2=dim, ch_int=dim, ch_out=dim)
        self.fu2 = EnhancedCrossAttentionHFF(ch_1=dim, ch_2=dim, ch_int=dim, ch_out=dim)
        #self.fu1 = HFF_block(ch_1=dim, ch_2=dim, r_2=4, ch_int=dim, ch_out=dim, drop_rate=HFF_dp)
        #self.fu2 = HFF_block(ch_1=dim, ch_2=dim, r_2=4, ch_int=dim, ch_out=dim, drop_rate=HFF_dp)
        self.compression1 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, bias=False),
            BatchNorm2d(channel // 4, momentum=bn_mom),
        )
        self.compression2 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, bias=False),
            BatchNorm2d(channel // 4, momentum=bn_mom),
        )
        self.compression3 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, bias=False),
            BatchNorm2d(channel // 4, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, bias=False),
            BatchNorm2d(channel // 4, momentum=bn_mom),
        )
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = self.fu1(self.compression2(self.FC11(x)), self.compression2(self.FC12(x)), self.FC13(x))  # L G

        x1 = self.FC1(F.relu(x1))

        # x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3
        # x21 = self.fuse21(self.FC21(x), self.FC22(x))
        # x22 = self.fuse22(self.FC22(x), self.FC23(x))
        # x23 = self.fuse23(self.FC21(x), self.FC23(x))
        # x2 = (x21 + x22 + x23) / 3
        x2 = self.fu2(self.compression3(self.FC21(x)), self.compression4(self.FC22(x)), self.FC23(x))
        x2 = self.FC2(F.relu(x2))

        # Use iAFF to fuse features: first x with x1, then the result with x2
        # out1 = self.fuse1(x, x1)
        # out2 = self.fuse2(x, x2)
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        # self.fu = HFF_block(ch_1=high_dim, ch_2=high_dim, r_2=16, ch_int=high_dim, ch_out=high_dim, drop_rate=0.)
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        #  = self.fu(W_y, x_h, None)

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        # z = self.fu(W_y, x_h, None)
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z



class CMFN(nn.Module):
    def __init__(self, class_num, dataset, arch='resnet50', planes=64):
        super(CMFN, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset
        self.layer3 = nn.Sequential(
            HCE(planes, planes * 2, planes * 4, dilation=[2, 3, 5]),
            # MFACB(planes * 4, planes * 4, planes * 4, dilation=[2, 3, 3]),
            HCE(planes * 4, planes * 4, planes, dilation=[2, 3, 5]),
        )
        self.layer4 = nn.Sequential(
            HCE(planes * 4, planes * 4, planes * 4, dilation=[2, 3, 5]),
            # MFACB(planes * 4, planes * 4, planes * 4, dilation=[2, 3, 5]),
            # MFACB(planes * 4, planes * 4, planes * 4, dilation=[2, 3, 5]),
        )
        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            pool_dim = 1024

            self.DEE = Multi_FUE(channel=512, dim=128,group_ratio=1,num_heads=16,window_size=8, shift_size=1,mlp_ratio=4)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
        else:
            pool_dim = 2048
            #self.DEE = DEE_module(1024)
            self.DEE = Multi_FUE(channel=1024,dim=256,group_ratio=1,num_heads=16,window_size=8,shift_size=1,mlp_ratio=4)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.MFA3 = MFA_block(1024, 512, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 4, 1, stride=1, dilation=5)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, dilation=dilation))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            # print("visible:", x1)
            x1 = self.visible_module(x1)
            # print("thermal:", x2)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:

            x = self.visible_module(x1)
        elif modal == 2:

            x = self.thermal_module(x2)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        # **c
        x_ = self.layer3(x_)

        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        # **

        x_ = self.layer4(x_)

        x_ = self.MFA2(x, x_)

        if self.dataset == 'regdb':  # For regdb dataset, we remove the MFA3 block and layer4.
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer3(x_)
        else:
            x = self.base_resnet.base.layer3(x_)

            # **
            x_ = self.layer5(x_)

            x_ = self.MFA3(x, x_)
            x_ = self.DEE(x_)

            x = self.base_resnet.base.layer4(x_)

        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))

            return x_pool, self.classifier(feat), loss_ort
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
