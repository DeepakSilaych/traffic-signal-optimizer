import torch
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)
        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = torch.relu(x)
        out = torch.relu(self.conv1(out))
        out = self.conv2(out)
        return out + self.shortcut(x)


class RecalibrationBlock(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(channels, height, width))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        return torch.sum(x * self.weights.unsqueeze(0), dim=1, keepdim=True)


class ClosenessComponent(nn.Module):
    def __init__(self, in_channels, T_c, height, width, nb_filters=32, num_res_units=2):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, nb_filters, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.T_new = T_c - 2
        self.reshaped_channels = nb_filters * self.T_new

        res_layers = [ResUnit(self.reshaped_channels, nb_filters)]
        for _ in range(num_res_units - 1):
            res_layers.append(ResUnit(nb_filters, nb_filters))
        self.res_stack = nn.Sequential(*res_layers)

        self.rc_block = RecalibrationBlock(nb_filters, height, width)

    def forward(self, x):
        out = torch.relu(self.conv3d(x))
        b, c, t, h, w = out.size()
        out = out.view(b, c * t, h, w)
        out = self.res_stack(out)
        return self.rc_block(out)


class WeeklyComponent(nn.Module):
    def __init__(self, in_channels, T_w, height, width, nb_filters=32):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, nb_filters, kernel_size=(T_w, 1, 1))
        self.rc_block = RecalibrationBlock(nb_filters, height, width)

    def forward(self, x):
        out = torch.relu(self.conv3d(x))
        b, c, t, h, w = out.size()
        out = out.view(b, c * t, h, w)
        return self.rc_block(out)


class ST3DNet(nn.Module):
    def __init__(self, in_channels=1, T_c=6, T_w=4, height=18, width=18, nb_filters=32, num_res_units=2):
        super().__init__()
        self.closeness = ClosenessComponent(in_channels, T_c, height, width, nb_filters, num_res_units)
        self.weekly = WeeklyComponent(in_channels, T_w, height, width, nb_filters)

        self.W_fc = nn.Parameter(torch.empty(1, height, width))
        self.W_fw = nn.Parameter(torch.empty(1, height, width))
        nn.init.xavier_uniform_(self.W_fc)
        nn.init.xavier_uniform_(self.W_fw)

    def forward(self, x_c, x_w):
        out_c = self.closeness(x_c)
        out_w = self.weekly(x_w)
        return torch.tanh(out_c * self.W_fc + out_w * self.W_fw)
