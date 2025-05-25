from tqdm.auto import tqdm
from losses import time_domain_l1
import torch
import torch.nn as nn

SAMPLE_RATE = 16000

def external_normalize(audio, eps=1e-3):
    mono = audio.mean(dim=1, keepdim=True)
    std = mono.std(dim=-1, keepdim=True)
    normalized = audio / (std + eps)
    return normalized, std

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, writer, global_step, LOAD_AUGS=False):
    try:
        if LOAD_AUGS:
            from augmentation import Remix, RevEcho, BandMask, Shift
            augmentations = nn.Sequential(
                Remix(),
                BandMask(maxwidth=0.2, bands=120, sample_rate=SAMPLE_RATE),
                Shift(same=True),
                RevEcho(proba=0.5, initial=0.3, rt60=(0.3, 1.3),
                        first_delay=(0.01, 0.03), repeat=3, jitter=0.1,
                        keep_clean=0.1, sample_rate=SAMPLE_RATE)
            )
            print("Augmentations loaded.")
        else:
            augmentations = None
            print("Augmentations not loaded.")
    except ImportError:
        augmentations = None
        print("Augmentation modules not found; proceeding without augmentations.")
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for i, (noisy, clean) in progress_bar:
        noisy = noisy.to(device)
        clean = clean.to(device)
        if augmentations is not None:
            noise = noisy - clean
            sources = torch.stack([noise, clean])
            sources = augmentations(sources)
            noise, clean = sources[0], sources[1]
            noisy = noise + clean

        normalized_noisy, std = external_normalize(noisy)
        optimizer.zero_grad()
        enhanced = model(normalized_noisy) * std

        total_loss = criterion(enhanced, clean)
        l1_loss = time_domain_l1(enhanced, clean)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item() * noisy.size(0)
        running_l1 += l1_loss.item() * noisy.size(0)

        writer.add_scalars("Loss/Iteration", 
                           {"Train Total": total_loss.item(), "Train L1": l1_loss.item()},
                           global_step)
        global_step += 1
        progress_bar.set_postfix(total_loss=total_loss.item(), l1_loss=l1_loss.item())
    scheduler.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_l1 = running_l1 / len(dataloader.dataset)
    return epoch_loss, epoch_l1, global_step

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_l1 = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False)
    with torch.no_grad():
        for i, (noisy, clean) in progress_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            normalized_noisy, std = external_normalize(noisy)
            enhanced = model(normalized_noisy) * std
            loss = criterion(enhanced, clean)
            l1_loss = time_domain_l1(enhanced, clean)
            running_loss += loss.item() * noisy.size(0)
            running_l1 += l1_loss.item() * noisy.size(0)
            progress_bar.set_postfix(total_loss=loss.item(), l1_loss=l1_loss.item())
    val_loss = running_loss / len(dataloader.dataset)
    val_l1 = running_l1 / len(dataloader.dataset)
    return val_loss, val_l1