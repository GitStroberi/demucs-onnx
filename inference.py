import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import sounddevice as sd

from model_def import (
    SAMPLE_RATE, FRAME_LENGTH,
    CausalDemucsSplit,
    DemucsEncoderONNX,
    DemucsDecoderONNX
)

##############################
# Global Hyperparameters & Parameters
##############################
DRY_MIX = 0.0  # fraction of dry (original) audio in the final mix.
SAMPLE_RATE = 16000
print("Frame length (samples):", FRAME_LENGTH)

##############################
# External Normalization Function
##############################
def external_normalize_np(audio, eps=1e-3):
    """
    Normalize the audio externally.
    Audio is expected to be a NumPy array of shape (B, 1, T).
    Returns:
      normalized_audio: audio divided by (std + eps)
      std: computed per-sample standard deviation (used later for re-scaling)
    """
    mono = np.mean(audio, axis=1, keepdims=True)
    std = np.std(mono, axis=-1, keepdims=True)
    normalized = audio / (std + eps)
    return normalized, std

##############################
# Load Trained Model & Create ONNX Sessions
##############################
device = torch.device("cpu")
model = CausalDemucsSplit(chin=1, chout=1, hidden=48, depth=5, kernel_size=8,
                          stride=4, causal=True, resample=4, normalize=False, glu=True).to(device)
# Adjust the following path as needed.
checkpoint_path = "demucs_model_finetune.pth"
assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found!"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Trained split model loaded (for LSTM processing).")

##########################
# Export Encoder & Decoder to ONNX 
##########################
encoder_onnx_module = DemucsEncoderONNX(model).to(device)
decoder_onnx_module = DemucsDecoderONNX(model).to(device)

dummy_input = torch.randn(1, 1, FRAME_LENGTH).to(device)
encoder_output_names = []
encoder_output_names.append("latent")
for i in range(1, model.depth+1):
    encoder_output_names.append(f"skip{i}")
encoder_onnx_path = "demucs_encoder.onnx"
torch.onnx.export(encoder_onnx_module, dummy_input, encoder_onnx_path,
                  opset_version=9,
                  input_names=["audio"],
                  output_names=encoder_output_names,
                  dynamic_axes=None)
print("Encoder exported to ONNX:", encoder_onnx_path)

# Prepare dummy output for decoder export.
with torch.no_grad():
    dummy_encoder_outputs = encoder_onnx_module(dummy_input)
    # Process dummy latent through LSTM:
    latent_dummy = dummy_encoder_outputs[0]
    latent_tensor = latent_dummy.float().to(device)
    latent_seq = latent_tensor.permute(0, 2, 1)
    lstm_out, _ = model.lstm(latent_seq)
    lstm_out = lstm_out.permute(0, 2, 1)
    # Combine LSTM output with original latent
    latent_for_decoder = lstm_out + latent_tensor
    # Assemble dummy decoder inputs: first latent_for_decoder, then each skip output (as Tensors)
    dummy_decoder_inputs = (latent_for_decoder,) + tuple(dummy_encoder_outputs[1:])
decoder_onnx_path = "demucs_decoder.onnx"
decoder_input_names = ["latent_out"]
for i in range(1, model.depth+1):
    decoder_input_names.append(f"skip{i}")
torch.onnx.export(decoder_onnx_module, dummy_decoder_inputs, decoder_onnx_path,
                  opset_version=9,
                  input_names=decoder_input_names,
                  output_names=["enhanced_audio"],
                  dynamic_axes=None)
print("Decoder exported to ONNX:", decoder_onnx_path)


encoder_sess = ort.InferenceSession(encoder_onnx_path)
decoder_sess = ort.InferenceSession(decoder_onnx_path)
encoder_input_name = encoder_sess.get_inputs()[0].name
decoder_input_names = [inp.name for inp in decoder_sess.get_inputs()]
decoder_output_name = decoder_sess.get_outputs()[0].name
print("Encoder and decoder ONNX sessions created.")

lstm_state = None

##############################
# Real-Time Audio Processing Callback
##############################
def audio_callback(indata, outdata, frames, time_info, status):
    global lstm_state
    if status:
        print("Status:", status)
        
    noisy_audio = np.expand_dims(indata.T, axis=0).astype(np.float32)  # shape: (1, 1, frames)
    
    normalized_audio, std = external_normalize_np(noisy_audio)
    
    # 1. Run Encoder ONNX model.
    enc_inputs = {encoder_input_name: normalized_audio}
    enc_outputs = encoder_sess.run(None, enc_inputs)
    latent_np = enc_outputs[0]  # expected shape: (1, C, T_enc)
    skip_list_np = enc_outputs[1:]
    
    # 2. Process latent output with the LSTM (PyTorch, CPU).
    latent_tensor = torch.from_numpy(latent_np).float()  # shape: (1, C, T_enc)
    latent_seq = latent_tensor.permute(0, 2, 1)  # shape: (1, T_enc, C)
    with torch.no_grad():
        lstm_out, lstm_state = model.lstm(latent_seq, lstm_state)
    lstm_out = lstm_out.permute(0, 2, 1)
    # Combine LSTM output with original latent
    combined_latent = (lstm_out + latent_tensor).cpu().numpy()
    
    # 3. Prepare inputs for Decoder ONNX.
    dec_inputs = { decoder_input_names[0]: combined_latent }
    for i, skip in enumerate(skip_list_np):
        dec_inputs[decoder_input_names[i+1]] = skip
        
    # 4. Run Decoder ONNX model.
    dec_output = decoder_sess.run([decoder_output_name], dec_inputs)[0]  # shape: (1, 1, FRAME_LENGTH)
    
    # 5. Un-normalize the enhanced output using the computed std.
    enhanced_audio = dec_output * std
    enhanced_audio = enhanced_audio.reshape(frames, 1)
    
    # 6. Mix the original (dry) audio with the enhanced (wet) audio.
    # DRY_MIX=0.0 → fully enhanced; DRY_MIX=1.0 → completely original.
    final_audio = DRY_MIX * indata + (1 - DRY_MIX) * enhanced_audio
    outdata[:] = final_audio

##############################
# Main: Start Real-Time Streaming Inference
##############################
if __name__ == '__main__':
    print("Starting real-time streaming denoising with split model...")
    # Audio stream parameters:
    stream = sd.Stream(samplerate=SAMPLE_RATE,
                       blocksize=FRAME_LENGTH,
                       dtype='float32',
                       channels=1,
                       callback=audio_callback)
    try:
        with stream:
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("Streaming stopped.")
