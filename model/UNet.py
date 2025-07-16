import torch
import torch.nn as nn
import torch.nn.functional as F

# CONVOLUTION
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    # 2D Convolution using a 3x3 kernel
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)


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

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(out_channels * 2, out_channels)
        else:  # add
            self.conv1 = conv3x3(out_channels, out_channels)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), dim=1)
        else:  # add
            x = from_up + from_down
        x_skip = self.conv1(x)
        x = F.relu(self.conv2(x_skip))
        x = F.relu(self.conv3(x) + x_skip)  # residual add
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_bins=32, out_mode='mean',
                 merge_mode='concat', depth=4, start_filters=64,
                 mode='hist'):
        """
        Args:
            in_channels: number of input color channels (usually 3)
            n_bins: number of histogram bins (only relevant if mode='hist')
            out_mode: 'mean' or 'distribution'
            merge_mode: 'add' (residual) or 'concat'
            depth: number of downsampling layers
            start_filters: number of filters in first conv block
            mode: 'hist' or 'img' input type
        """
        super(UNet, self).__init__()
        assert merge_mode in ('add', 'concat'), "merge_mode must be 'add' or 'concat'"
        assert mode in ('hist', 'img'), "mode must be 'hist' or 'img'"

        self.out_mode = out_mode
        self.n_bins = n_bins
        self.merge_mode = merge_mode
        self.depth = depth
        self.start_filters = start_filters
        self.mode = mode

        # Determine input channels based on mode
        if self.mode == 'hist':
            # self.input_channels = in_channels * n_bins
            # include mean + variance: 
            self.input_channels = in_channels * (n_bins + 2)
        else:  # 'img' mode
            self.input_channels = in_channels

        # Encoder
        self.down_convs = nn.ModuleList()
        in_ch = self.input_channels
        for i in range(depth):
            out_ch = start_filters * (2 ** i)
            pooling = (i < depth - 1)
            self.down_convs.append(DownConv(in_ch, out_ch, pooling=pooling))
            in_ch = out_ch

        # Decoder
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            in_ch = start_filters * (2 ** (i + 1))
            out_ch = start_filters * (2 ** i)
            self.up_convs.append(UpConv(in_ch, out_ch, merge_mode=merge_mode))

        # Final conv layer
        if out_mode == 'mean':
            self.final = nn.Conv2d(start_filters, 3, kernel_size=1)
        elif out_mode == 'distribution':
            self.final = nn.Conv2d(start_filters, 3 * n_bins, kernel_size=1)
        else:
            raise ValueError("Invalid out_mode. Use 'mean' or 'distribution'.")

    def forward(self, x):
        """
        Forward pass.
        Input shape depends on mode:
          - hist: x is (B, 3, bins, H, W)
          - img: x is (B, 3, H, W)
        Output:
          - mean image (B, 3, H, W) or distribution (B, 3, n_bins, H, W)
        """
        if self.mode == 'hist':
            B, C, bins, H, W = x.shape
            x = x.view(B, C * bins, H, W)

        else:  # 'img' mode
            # input is already (B, 3, H, W)
            pass

        encoder_outs = []

        # Encoder
        for down in self.down_convs:
            x, before_pool = down(x)
            encoder_outs.append(before_pool)

        # Decoder
        for i, up in enumerate(self.up_convs):
            skip = encoder_outs[-(i + 2)]
            x = up(skip, x)

        out = self.final(x)

        if self.out_mode == 'distribution':
            B, _, H, W = out.shape
            out = out.view(B, 3, self.n_bins, H, W)
            out = F.softmax(out, dim=2)  # softmax over bins

        return out
