# -----------------------------------------------------------------------------------
# 1. SETUP & VALID LENGTH COMPUTATION
# -----------------------------------------------------------------------------------
import os
import glob
import math
import random
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.auto import tqdm
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image
import numpy as np
from datetime import datetime

print("PyTorch version:", torch.__version__)
print("Torch Audio version:", torchaudio.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Global Hyperparameters
BATCH_SIZE = 32
SAMPLE_RATE = 16000
NUM_EPOCHS = 200         # Continue training for 200 additional epochs.
LEARNING_RATE = 3e-4      # Learning rate.
TARGET_AUDIO_LENGTH = 2.0 # in seconds

# Demucs design parameters
H = 48      # initial number of hidden channels.
L = 5       # number of encoder/decoder layers.
K = 8       # kernel size.
S_conv = 4  # stride.
RESAMPLE_FACTOR = 4  # upsample/downsample factor.

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

# -----------------------------------------------------------------------------------
# 2. DATA PREPARATION: MS-SNSD DATASET LOADING
# -----------------------------------------------------------------------------------
# For the MS-SNSD dataset, we assume the following structure:
#
# dataset/
#   CleanSpeech_training/    <-- Contains clean .wav files.
#   Noise_training/          <-- (May be used for augmentation if desired.)
#   NoisySpeech_training/    <-- Contains noisy versions of the clean files with 5 different SNR levels.
#
# Each clean file (e.g. clnsp1.wav) has noisy counterparts like:
# noisy1_SNRdb_0.0_clnsp1.wav, noisy1_SNRdb_10.0_clnsp1.wav, etc.

class MSSNSDDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=SAMPLE_RATE, target_length=VALID_LENGTH, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_num_samples = target_length

        # List all clean files.
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        self.pairs = []
        # For each clean file, find all corresponding noisy files.
        for clean_path in self.clean_files:
            base_clean = os.path.basename(clean_path)
            # Create a pattern that matches any noisy file ending with _<clean filename>
            pattern = os.path.join(noisy_dir, f"*_{base_clean}")
            noisy_matches = sorted(glob.glob(pattern))
            if len(noisy_matches) == 0:
                print(f"Warning: No matching noisy files for {base_clean}. Skipping.")
            for noisy_path in noisy_matches:
                self.pairs.append((noisy_path, clean_path))
        if len(self.pairs) == 0:
            raise ValueError("No paired files found. Check your dataset paths and naming conventions.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_waveform, sr_noisy = torchaudio.load(noisy_path)
        clean_waveform, sr_clean = torchaudio.load(clean_path)

        # Resample if necessary.
        if sr_noisy != self.sample_rate:
            resampler = T.Resample(sr_noisy, self.sample_rate)
            noisy_waveform = resampler(noisy_waveform)
        if sr_clean != self.sample_rate:
            resampler = T.Resample(sr_clean, self.sample_rate)
            clean_waveform = resampler(clean_waveform)

        # Ensure mono.
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)
        clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)

        # Synchronized crop (or pad) to the valid length.
        noisy_waveform, clean_waveform = self._synchronized_crop_or_pad(noisy_waveform, clean_waveform)

        if self.transform:
            noisy_waveform = self.transform(noisy_waveform)
            clean_waveform = self.transform(clean_waveform)
        return noisy_waveform, clean_waveform

    def _synchronized_crop_or_pad(self, noisy, clean):
        length = noisy.shape[-1]
        target = self.target_num_samples
        if length > target:
            start = random.randint(0, length - target)
            noisy = noisy[..., start:start + target]
            clean = clean[..., start:start + target]
        elif length < target:
            pad_amt = target - length
            noisy = F.pad(noisy, (0, pad_amt))
            clean = F.pad(clean, (0, pad_amt))
        return noisy, clean

DATA_ROOT = "/path/to/your/dataset/location/" # Adjust this path to your dataset location.
clean_train_dir = os.path.join(DATA_ROOT, "CleanSpeech_training")
noisy_train_dir = os.path.join(DATA_ROOT, "NoisySpeech_training")

# Create full dataset and then an 80/20 train/validation split.
full_dataset = MSSNSDDataset(noisy_train_dir, clean_train_dir)
from torch.utils.data import random_split
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_subset, val_subset = random_split(full_dataset, [train_len, val_len])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

print("Training pairs:", len(train_subset))
print("Validation pairs:", len(val_subset))

from model_def import CausalDemucsSplit

# Instantiate the model.
model = CausalDemucsSplit(chin=1, chout=1, hidden=H, depth=L, kernel_size=K, stride=S_conv,
                     causal=True, resample=RESAMPLE_FACTOR, normalize=False, glu=True).to(device)
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Define an external normalization function.
def external_normalize(audio, eps=1e-3):
    mono = audio.mean(dim=1, keepdim=True)
    std = mono.std(dim=-1, keepdim=True)
    normalized = audio / (std + eps)
    return normalized, std

# -----------------------------------------------------------------------------------
# 4. ONNX EXPORT TEST
# -----------------------------------------------------------------------------------
dummy_input = torch.randn(1, 1, FRAME_LENGTH).to(device)
model.eval()
export_model = model.module if isinstance(model, nn.DataParallel) else model
try:
    torch.onnx.export(
        export_model, dummy_input, "denoiser_causal.onnx",
        opset_version=9,
        input_names=["noisy_audio"],
        output_names=["denoised_audio"],
        dynamic_axes=None
    )
    print("ONNX export successful!")
except Exception as e:
    print("ONNX export failed:", e)

from losses import CombinedLoss

criterion = CombinedLoss(
    fft_sizes=[512, 1024, 2048],
    hop_sizes=[50, 120, 240],
    win_lengths=[240, 600, 1200],
    window="hann_window",
    factor_sc=0.1,
    factor_mag=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Convert spectrograms to images.
def plot_spectrogram_to_image(waveform, title, sample_rate=SAMPLE_RATE):
    plt.figure(figsize=(12, 4))
    plt.specgram(waveform.squeeze().cpu().numpy(), NFFT=512, Fs=sample_rate, noverlap=256, cmap='viridis')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image).transpose(2, 0, 1)  # Convert HWC to CHW.
    return image

# -----------------------------------------------------------------------------------
# 6. TRAINING & VALIDATION LOOP (with TensorBoard logging)
# -----------------------------------------------------------------------------------
# Create TensorBoard SummaryWriter instance.
current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
log_dir = f"runs/denoiser/experiment_{current_time}"
writer = SummaryWriter(log_dir)

global_step = 0

# --- Load pretrained model before continuing training ---
pretrained_model_path = "demucs_model.pth"  # <-- Update this path!
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))
    print("Loaded pretrained model for continued training.")

from training import train_one_epoch, validate_one_epoch

train_losses = []
val_losses = []
train_l1_losses = []
val_l1_losses = []
best_val_loss = float("inf")
best_model_path = "demucs_model_finetune.pth"

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss, train_l1, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, writer, global_step, LOAD_AUGS=False)
    val_loss, val_l1 = validate_one_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_l1_losses.append(train_l1)
    val_l1_losses.append(val_l1)
    print(f"Train Loss: {train_loss:.4f} | Train L1: {train_l1:.4f} | Val Loss: {val_loss:.4f} | Val L1: {val_l1:.4f}")

    writer.add_scalars("Loss/Epoch",
                       {"Train Total": train_loss,
                        "Train L1": train_l1,
                        "Validation Total": val_loss,
                        "Validation L1": val_l1},
                       epoch)

    with torch.no_grad():
        sample_noisy, sample_clean = val_subset[0]
        sample_noisy = sample_noisy.unsqueeze(0).to(device)
        sample_clean = sample_clean.unsqueeze(0).to(device)
        norm_noisy, sample_std = external_normalize(sample_noisy)
        sample_enhanced = model(norm_noisy) * sample_std
        writer.add_audio("Audio/Noisy", sample_noisy[0], epoch, sample_rate=SAMPLE_RATE)
        writer.add_audio("Audio/Clean", sample_clean[0], epoch, sample_rate=SAMPLE_RATE)
        writer.add_audio("Audio/Enhanced", sample_enhanced[0].cpu(), epoch, sample_rate=SAMPLE_RATE)
        noisy_spec = plot_spectrogram_to_image(sample_noisy[0], "Noisy Spectrogram")
        clean_spec = plot_spectrogram_to_image(sample_clean[0], "Clean Spectrogram")
        enhanced_spec = plot_spectrogram_to_image(sample_enhanced[0].cpu(), "Enhanced Spectrogram")
        writer.add_image("Spectrogram/Noisy", noisy_spec, epoch)
        writer.add_image("Spectrogram/Clean", clean_spec, epoch)
        writer.add_image("Spectrogram/Enhanced", enhanced_spec, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with val_loss: {best_val_loss:.4f}")

writer.close()

# -----------------------------------------------------------------------------------
# 7. LOSS PLOT SAVING
# -----------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss (Total)')
plt.plot(val_losses, label='Validation Loss (Total)')
plt.plot(train_l1_losses, label='Training L1 Loss')
plt.plot(val_l1_losses, label='Validation L1 Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot_ms_snsd.png")
plt.show()