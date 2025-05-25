# -----------------------------------------------------------------------------------
# 1. SETUP & VALID LENGTH COMPUTATION
# -----------------------------------------------------------------------------------
import os
import glob
import math
import torch
import random
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

print("PyTorch version:", torch.__version__)
print("Torch Audio version:", torchaudio.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Global Hyperparameters
BATCH_SIZE = 64
SAMPLE_RATE = 16000
NUM_EPOCHS = 400         # Number of training epochs.
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
    convolution, and downsampling operations the output length equals the input.
    """
    L_val = math.ceil(base_length * resample)
    for _ in range(depth):
        L_val = math.ceil((L_val - kernel_size) / stride) + 1
        L_val = max(L_val, 1)
    for _ in range(depth):
        L_val = (L_val - 1) * stride + kernel_size
    valid_length = int(math.ceil(L_val / resample))
    return valid_length

# Compute a valid length based on the target audio length.
BASE_LENGTH = int(SAMPLE_RATE * TARGET_AUDIO_LENGTH)
VALID_LENGTH = compute_valid_length(BASE_LENGTH, L, K, S_conv, RESAMPLE_FACTOR)
print("Valid input length for the network:", VALID_LENGTH)

# We'll also use VALID_LENGTH for the dummy input in ONNX export.
FRAME_LENGTH = VALID_LENGTH

# -----------------------------------------------------------------------------------
# 2. DATA PREPARATION (Dataset Crops to VALID_LENGTH)
# -----------------------------------------------------------------------------------
class ValentiniDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=SAMPLE_RATE, target_length=VALID_LENGTH, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_num_samples = target_length  # use the computed valid length

        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch between number of noisy and clean files!"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]
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

        # Synchronize random cropping (or padding) so that both waveforms are aligned.
        noisy_waveform, clean_waveform = self._synchronized_crop_or_pad(noisy_waveform, clean_waveform)

        if self.transform:
            noisy_waveform = self.transform(noisy_waveform)
            clean_waveform = self.transform(clean_waveform)
        return noisy_waveform, clean_waveform

    def _synchronized_crop_or_pad(self, noisy, clean):
        # Assume both have the same length
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
noisy_train_dir = os.path.join(DATA_ROOT, "noisy_trainset_56spk_wav")
clean_train_dir = os.path.join(DATA_ROOT, "clean_trainset_56spk_wav")
noisy_test_dir = os.path.join(DATA_ROOT, "noisy_testset_wav")
clean_test_dir = os.path.join(DATA_ROOT, "clean_testset_wav")

train_dataset = ValentiniDataset(noisy_train_dir, clean_train_dir)
test_dataset = ValentiniDataset(noisy_test_dir, clean_test_dir)

from torch.utils.data import random_split
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_subset, val_subset = random_split(train_dataset, [train_len, val_len])

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

print("Training samples:", len(train_subset))
print("Validation samples:", len(val_subset))
print("Testing samples:", len(test_dataset))

# -----------------------------------------------------------------------------------
# 3. MODEL DEFINITION: ONNX-Compatible CausalDemucs (Split into Encoder, LSTM, Decoder)
# -----------------------------------------------------------------------------------
# Precompute constant sinc kernels for upsampling/downsampling.

from model_def import CausalDemucsSplit

model = CausalDemucsSplit(chin=1, chout=1, hidden=H, depth=L, kernel_size=K, stride=S_conv,
                          causal=True, resample=RESAMPLE_FACTOR, normalize=False, glu=True).to(device)

print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Define an external normalization function.
def external_normalize(audio, eps=1e-3):
    """
    Normalize the audio externally.
    Returns:
      normalized_audio: audio divided by (std + eps)
      std: computed per-sample standard deviation (used later for re-scaling)
    """
    mono = audio.mean(dim=1, keepdim=True)
    std = mono.std(dim=-1, keepdim=True)
    normalized = audio / (std + eps)
    return normalized, std


# -----------------------------------------------------------------------------------
# 4. ONNX EXPORT TEST (Opset 9)
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

# -----------------------------------------------------------------------------------
# 5. LOSS FUNCTIONS AND AUXILIARY FUNCTIONS
# -----------------------------------------------------------------------------------
from losses import time_domain_l1, CombinedLoss

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

# -----------------------------------------------------------------------------------
# Helper Function: Convert a spectrogram plot to an image array for TensorBoard.
# -----------------------------------------------------------------------------------
def plot_spectrogram_to_image(waveform, title, sample_rate=SAMPLE_RATE):
    """
    Plots a spectrogram of the provided waveform and returns the image as a numpy array in CHW format.
    """
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
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    return image

# -----------------------------------------------------------------------------------
# 6. TRAINING & VALIDATION LOOP
# -----------------------------------------------------------------------------------
from datetime import datetime
current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
log_dir = f"runs/denoiser/experiment_{current_time}"
writer = SummaryWriter(log_dir)
global_step = 0

from training import train_one_epoch, validate_one_epoch

train_losses = []
val_losses = []
train_l1_losses = []
val_l1_losses = []
best_val_loss = float("inf")
best_model_path = "demucs_model.pth"

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model.")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss, train_l1, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, writer, global_step, LOAD_AUGS=True)
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
        sample_noisy, sample_clean = test_dataset[0]
        sample_noisy = sample_noisy.unsqueeze(0).to(device)
        sample_clean = sample_clean.unsqueeze(0).to(device)
        norm_noisy, sample_std = external_normalize(sample_noisy)
        sample_enhanced = model(norm_noisy) * sample_std
        writer.add_audio("Audio/Noisy", sample_noisy[0], epoch, sample_rate=SAMPLE_RATE)
        writer.add_audio("Audio/Clean", sample_clean[0], epoch, sample_rate=SAMPLE_RATE)
        writer.add_audio("Audio/Enhanced", sample_enhanced[0].cpu(), epoch, sample_rate=SAMPLE_RATE)
        noisy_spec = plot_spectrogram_to_image(sample_noisy[0], "Noisy Spectrogram", SAMPLE_RATE)
        clean_spec = plot_spectrogram_to_image(sample_clean[0], "Clean Spectrogram", SAMPLE_RATE)
        enhanced_spec = plot_spectrogram_to_image(sample_enhanced[0].cpu(), "Enhanced Spectrogram", SAMPLE_RATE)
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
plt.savefig("loss_plot_valentini.png")
plt.show()