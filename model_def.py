# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# author: adefossez

import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# Demucs design parameters
H = 48      # initial number of hidden channels.
L = 5       # number of encoder/decoder layers.
K = 8       # kernel size.
S_conv = 4  # stride.
RESAMPLE_FACTOR = 4  # upsample/downsample factor.

SAMPLE_RATE = 16000
TARGET_AUDIO_LENGTH = 2 # in seconds, change as needed.

# --- Compute a valid length ---
def compute_valid_length(base_length, depth, kernel_size, stride, resample):
    """
    Compute the nearest valid input length such that after the upsampling,
    convolution, downsampling operations the output length equals the input length.
    """
    L_val = math.ceil(base_length * resample)
    for _ in range(depth):
        L_val = math.ceil((L_val - kernel_size) / stride) + 1
        L_val = max(L_val, 1)
    for _ in range(depth):
        L_val = (L_val - 1) * stride + kernel_size
    valid_length = int(math.ceil(L_val / resample))
    return valid_length

BASE_LENGTH = int(SAMPLE_RATE * TARGET_AUDIO_LENGTH)
VALID_LENGTH = compute_valid_length(BASE_LENGTH, L, K, S_conv, RESAMPLE_FACTOR)
print("Valid input length for the network:", VALID_LENGTH)
FRAME_LENGTH = VALID_LENGTH

UPSAMPLE_ZEROS = 56
_kernel_up = None
_kernel_down = None

def get_upsample_kernel():
    global _kernel_up
    if _kernel_up is None:
        t = torch.linspace(-UPSAMPLE_ZEROS + 0.5, UPSAMPLE_ZEROS - 0.5, 2 * UPSAMPLE_ZEROS)
        sinc_filter = torch.where(t == 0, torch.tensor(1.0), torch.sin(math.pi * t) / (math.pi * t))
        window = torch.hann_window(4 * UPSAMPLE_ZEROS + 1, periodic=False)[1::2]
        kernel = (sinc_filter * window).view(1, 1, -1)
        _kernel_up = kernel
    return _kernel_up

def get_downsample_kernel():
    global _kernel_down
    if _kernel_down is None:
        t = torch.linspace(-UPSAMPLE_ZEROS + 0.5, UPSAMPLE_ZEROS - 0.5, 2 * UPSAMPLE_ZEROS)
        sinc_filter = torch.where(t == 0, torch.tensor(1.0), torch.sin(math.pi * t) / (math.pi * t))
        window = torch.hann_window(4 * UPSAMPLE_ZEROS + 1, periodic=False)[1::2]
        kernel = (sinc_filter * window).view(1, 1, -1)
        _kernel_down = kernel
    return _kernel_down

def upsample2(x):
    kernel = get_upsample_kernel().to(x.device).to(x.dtype)
    B, C, T = x.shape
    filtered = F.conv1d(x.view(-1, 1, T), kernel, padding=UPSAMPLE_ZEROS)[..., 1:]
    filtered = filtered.view(B, C, -1)
    out = torch.stack((x, filtered), dim=3)  # shape: (B, C, T, 2)
    out = out.view(B, C, -1)  # shape: (B, C, 2*T)
    return out

def downsample2(x):
    kernel = get_downsample_kernel().to(x.device).to(x.dtype)
    B, C, T = x.shape
    if T % 2 == 1:
        x = F.pad(x, (0, 1))
        T += 1
    x_reshaped = x.view(B, C, T//2, 2)
    x_even = x_reshaped[..., 0]
    x_odd  = x_reshaped[..., 1]
    filtered = F.conv1d(x_odd.view(-1, 1, x_odd.shape[-1]), kernel, padding=UPSAMPLE_ZEROS)
    filtered = filtered.view(B, C, -1)[..., :-1]  # drop extra sample
    out = 0.5 * (x_even + filtered)
    return out

class CausalDemucsSplit(nn.Module):
    def __init__(self, chin=1, chout=1, hidden=H, depth=L, kernel_size=K, stride=S_conv,
                 causal=True, resample=RESAMPLE_FACTOR, growth=2.0, max_hidden=10_000,
                 normalize=False, glu=True):
        """
        Causal Demucs model split into encoder, LSTM, and decoder.
        The encoder and decoder parts are ONNX-compatible (fixed input size).
        The LSTM bottleneck remains in PyTorch and can be executed on the CPU.
        """
        super().__init__()
        assert causal, "Only causal mode is supported for ONNX export."
        self.normalize = normalize
        self.floor = 1e-3
        self.resample = resample
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels = []
        in_channels = chin
        ch = hidden
        activation = nn.GLU(dim=1) if glu else nn.ReLU()
        # Build encoder: record the output channels for each layer.
        for i in range(depth):
            out_channels = min(int(ch), max_hidden)
            encoder_channels.append(out_channels)
            enc_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels * 2, kernel_size=1),
                activation
            )
            self.encoder.append(enc_block)
            in_channels = out_channels
            ch *= growth
        # LSTM bottleneck with input size equal to the output of the final encoder layer.
        self.lstm = nn.LSTM(input_size=encoder_channels[-1], hidden_size=encoder_channels[-1],
                            num_layers=2, batch_first=True, bidirectional=False)
        # Build decoder in reverse order.
        for i in range(depth - 1, -1, -1):
            if i == 0:
                out_ch = chout
            else:
                out_ch = encoder_channels[i - 1]
            dec_block = nn.Sequential(
                nn.Conv1d(encoder_channels[i], encoder_channels[i] * 2, kernel_size=1),
                activation,
                nn.ConvTranspose1d(encoder_channels[i], out_ch, kernel_size, stride=stride, padding=0)
            )
            if i != 0:
                dec_block.add_module("relu", nn.ReLU())
            self.decoder.append(dec_block)
        self.encoder_channels = encoder_channels
        self.depth = depth

    def encode(self, audio):
        """
        Process input audio through the encoder.
        Returns:
           latent: output of the final encoder layer (to be fed into the LSTM)
           skips: list of all encoder outputs (skip connections) in order
           orig_len: original input length (or padded length)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        orig_len = audio.shape[-1]
        x = audio
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = x / (std + self.floor)
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        latent = x  # final encoder output
        return latent, skips, orig_len

    def process_bottleneck(self, latent):
        """
        Process the encoder latent representation through the LSTM.
        """
        x = latent.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        return x

    def decode(self, x, skip_list, orig_len):
        """
        Decode the LSTM-processed latent using the stored skip connections.
        """
        skips = skip_list.copy()
        for dec in self.decoder:
            if skips:
                skip = skips.pop()
                assert skip.shape[-1] == x.shape[-1], "Mismatch in feature lengths between skip and decoder."
                x = x + skip
                x = dec(x)
            else:
                raise ValueError("Not enough skip connections for decoding.")
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        assert x.shape[-1] == orig_len, f"Output length {x.shape[-1]} does not match input length {orig_len}"
        return x

    def forward(self, audio):
        """
        Full end-to-end forward pass: encode -> LSTM bottleneck -> decode.
        """
        latent, skip_list, orig_len = self.encode(audio)
        processed_latent = self.process_bottleneck(latent)
        enhanced = self.decode(processed_latent, skip_list, orig_len)
        return enhanced
    
##########################
# --- ONNX Wrapper Classes ---
##########################
# These wrappers allow exporting the encoder and decoder separately.
class DemucsEncoderONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, audio):
        # Return latent and all skip connections as separate outputs.
        latent, skips, _ = self.model.encode(audio)
        # Return a tuple: first element is latent, then each skip.
        outs = [latent] + skips
        return tuple(outs)

class DemucsDecoderONNX(nn.Module):
    def __init__(self, model, frame_length=FRAME_LENGTH):
        super().__init__()
        self.model = model
        self.frame_length = frame_length
    def forward(self, *inputs):
        # Expect inputs: first is the LSTM output (to be added with latent) then the skip connections.
        latent_out = inputs[0]
        skip_list = list(inputs[1:])
        return self.model.decode(latent_out, skip_list, self.frame_length)