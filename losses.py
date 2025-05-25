# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------------
# 5. LOSS FUNCTIONS AND AUXILIARY FUNCTIONS
# -----------------------------------------------------------------------------------
def time_domain_l1(y_pred, y_true):
    return nn.functional.l1_loss(y_pred, y_true)

def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(
        x, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
        window=window, center=True, return_complex=False
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    mag = torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7))
    mag = mag.transpose(1, 2)
    return mag

class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_mag, y_mag):
        return nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))

class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
    def forward(self, x, y):
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, ss, wl, window) for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for loss_fn in self.stft_losses:
            sc_l, mag_l = loss_fn(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return self.factor_sc * sc_loss, self.factor_mag * mag_loss

class CombinedLoss(nn.Module):
    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[50, 120, 240],
                 win_lengths=[240, 600, 1200],
                 window="hann_window",
                 factor_sc=1.0,
                 factor_mag=1.0):
        super().__init__()
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            factor_sc=factor_sc,
            factor_mag=factor_mag
        )
    def forward(self, pred, target):
        pred_2d = pred.squeeze(1)
        target_2d = target.squeeze(1)
        l1_term = time_domain_l1(pred, target)
        sc_loss, mag_loss = self.mrstft(pred_2d, target_2d)
        return l1_term + sc_loss + mag_loss