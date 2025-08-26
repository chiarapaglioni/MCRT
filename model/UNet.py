import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


# CONVOLUTIONS
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        padding=padding, 
        bias=bias)


def upconv2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


# DOWN PATH
class DownConv(nn.Module):
    """
    Residual Down Convolution Block with optional max pooling.
    3 conv layers with residual connection + pooling.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.pooling = pooling

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_skip = self.conv1(x)
        x = F.relu(self.conv2(x_skip))
        x = F.relu(self.conv3(x) + x_skip)  # residual add
        
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
    

# UP PATH
class UpConv(nn.Module):
    """
    Residual Up Convolution Block.
    Supports skip connection merge_mode: 'add' or 'concat'.
    Upsamples with ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        assert merge_mode in ('add', 'concat'), "merge_mode must be 'add' or 'concat'"
        assert up_mode in ('transpose',), "only 'transpose' up_mode supported for now"

        self.merge_mode = merge_mode

        self.upconv = upconv2x2(in_channels, out_channels)

        # merge mode
        if self.merge_mode == 'concat':     # CONCAT
            self.conv1 = conv3x3(out_channels * 2, out_channels)
        else:                               # ADD
            self.conv1 = conv3x3(out_channels, out_channels)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        # merge mode
        if self.merge_mode == 'concat':     # CONCAT
            x = torch.cat((from_up, from_down), dim=1)
        else:                               # ADD
            x = from_up + from_down
        
        x_skip = self.conv1(x)
        x = F.relu(self.conv2(x_skip))
        x = F.relu(self.conv3(x) + x_skip)  # residual add
        return x

class GapUNet(nn.Module):
    def __init__(self, in_channels=3, n_bins_input=16, n_bins_output=16, out_mode='mean',
                 merge_mode='concat', depth=4, start_filters=64, mode='hist',
                 grouped_rgb_head=True):
        super().__init__()
        assert merge_mode in ('add', 'concat')
        assert mode in ('hist', 'img')
        assert out_mode in ('mean', 'dist')

        self.out_mode = out_mode
        self.n_bins_input = n_bins_input
        self.n_bins_output = n_bins_output
        self.merge_mode = merge_mode
        self.depth = depth
        self.start_filters = start_filters
        self.mode = mode
        self.grouped_rgb_head = grouped_rgb_head

        # ---- Encoder ----
        in_ch = in_channels
        self.down_convs = nn.ModuleList()
        for i in range(depth):
            out_ch = start_filters * (2 ** i)
            pooling = (i < depth - 1)
            self.down_convs.append(DownConv(in_ch, out_ch, pooling=pooling))
            in_ch = out_ch

        # ---- Decoder ----
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            in_ch = start_filters * (2 ** (i + 1))
            out_ch = start_filters * (2 ** i)
            self.up_convs.append(UpConv(in_ch, out_ch, merge_mode=merge_mode))

        # ---- Heads ----
        # Features leaving the decoder have 'start_filters' channels
        feat_ch = start_filters

        if out_mode == 'mean':
            self.head = nn.Conv2d(feat_ch, 3, kernel_size=1, bias=True)

        else:  # 'dist' => want logits of shape (B, 3, n_bins_output, H, W)
            out_ch = 3 * n_bins_output

            if grouped_rgb_head:
                # Option A (strictly per-channel heads):
                # Make sure feature channels are divisible by 3 for groups=3.
                # If not, expand to 3*feat_ch first so each RGB gets its own copy.
                if feat_ch % 3 != 0:
                    # replicate features into three color groups
                    self.color_expand = nn.Conv2d(feat_ch, 3 * feat_ch, kernel_size=1, bias=True)
                    feat_ch_for_head = 3 * feat_ch
                else:
                    self.color_expand = None
                    feat_ch_for_head = feat_ch

                # Each color group sees its own slice of features and outputs n_bins each
                self.head = nn.Conv2d(feat_ch_for_head, out_ch, kernel_size=1, groups=3, bias=True)

            else:
                # Option B (shared features across RGB; a bit simpler and often works well):
                self.color_expand = None
                self.head = nn.Conv2d(feat_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        # ---- Encoder ----
        skips = []
        for down in self.down_convs:
            x, before_pool = down(x)
            skips.append(before_pool)

        # ---- Decoder ----
        for i, up in enumerate(self.up_convs):
            skip = skips[-(i + 2)]
            x = up(skip, x)

        feats = x  # last decoder features, shape: (B, start_filters, H, W)

        # ---- Head ----
        if self.out_mode == 'mean':
            out = self.head(feats)  # (B, 3, H, W)
            return out

        # 'dist'
        if hasattr(self, 'color_expand') and self.color_expand is not None:
            feats = self.color_expand(feats)  # -> (B, 3*start_filters, H, W)

        logits = self.head(feats)             # (B, 3*n_bins, H, W)
        B, _, H, W = logits.shape
        logits = logits.view(B, 3, self.n_bins_output, H, W)
        return logits
