import sys
import threading
import numpy as np
from scipy.signal import stft
import torch
import onnxruntime as ort
import sounddevice as sd
sd.default.latency = 'low'
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

from model_def import (
    SAMPLE_RATE,
    FRAME_LENGTH,
    CausalDemucsSplit
)

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


# ─── Model & ONNX Initialization ────────────────────────────────────────────────
device = torch.device("cpu")
model = CausalDemucsSplit(chin=1, chout=1, hidden=48, depth=5,
                          kernel_size=8, stride=4, causal=True,
                          resample=4, normalize=False, glu=True).to(device)
model.load_state_dict(torch.load("best_model_ms_snsd_split.pth", map_location=device))
model.eval()

encoder_sess = ort.InferenceSession("demucs_encoder.onnx")
decoder_sess = ort.InferenceSession("demucs_decoder.onnx")
enc_input_name = encoder_sess.get_inputs()[0].name
dec_input_names = [inp.name for inp in decoder_sess.get_inputs()]
dec_output_name = decoder_sess.get_outputs()[0].name

# Hold LSTM state across blocks
lstm_state = None

# ─── Signal Processing Helpers ───────────────────────────────────────────────────
def compute_spectrogram(audio, sr=SAMPLE_RATE, n_fft=512, hop_length=256):
    _, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    S_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
    return S_db  # shape: (freq_bins, time_frames)

def process_block(indata):
    """Return raw_block, enhanced_block, and enhanced_spectrogram."""
    global lstm_state

    # reshape to (1,1,frames)
    raw = indata.T[np.newaxis, ...].astype(np.float32)
    normalized, std = external_normalize_np(raw)

    # ONNX Encoder
    enc_outs = encoder_sess.run(None, {enc_input_name: normalized})
    latent_np, *skips = enc_outs

    # LSTM
    latent_t = torch.from_numpy(latent_np).float()
    seq = latent_t.permute(0, 2, 1)
    with torch.no_grad():
        lstm_out, lstm_state = model.lstm(seq, lstm_state)
    lstm_np = lstm_out.permute(0, 2, 1).cpu().numpy()
    combined = latent_np + lstm_np

    # ONNX Decoder
    dec_in = {dec_input_names[0]: combined}
    for i, skip in enumerate(skips):
        dec_in[dec_input_names[i+1]] = skip
    enhanced_np = decoder_sess.run([dec_output_name], dec_in)[0]

    # de-normalize & reshape
    enhanced = (enhanced_np * std).reshape(-1, 1)

    # spectrogram of enhanced
    spec = compute_spectrogram(enhanced.flatten())
    return raw.reshape(-1,1), enhanced, spec

# ─── Audio Worker ────────────────────────────────────────────────────────────────
class AudioWorker(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.muted = False
    
    raw_wave_ready = QtCore.pyqtSignal(np.ndarray)
    raw_spec_ready = QtCore.pyqtSignal(np.ndarray)
    enh_wave_ready = QtCore.pyqtSignal(np.ndarray)
    enh_spec_ready = QtCore.pyqtSignal(np.ndarray)

    @QtCore.pyqtSlot()
    def run(self):
        def callback(indata, outdata, frames, time_info, status):
            if status:
                print("Stream status:", status)
                
            if self.muted:
                # if muted, zero the output and still emit blank visuals
                outdata[:] = np.zeros_like(indata)
                self.raw_wave_ready.emit(indata.flatten())
                self.raw_spec_ready.emit(compute_spectrogram(indata.flatten()))
                self.enh_wave_ready.emit(np.zeros_like(indata).flatten())
                self.enh_spec_ready.emit(compute_spectrogram(np.zeros_like(indata).flatten()))
                return
            
            raw_blk, enh_blk, enh_spec = process_block(indata)
            raw_spec = compute_spectrogram(raw_blk.flatten())

            self.raw_wave_ready.emit(raw_blk.flatten())
            self.raw_spec_ready.emit(raw_spec)
            self.enh_wave_ready.emit(enh_blk.flatten())
            self.enh_spec_ready.emit(enh_spec)

            outdata[:] = enh_blk

        with sd.Stream(samplerate=SAMPLE_RATE,
                       blocksize=FRAME_LENGTH,
                       dtype='float32',
                       channels=1,
                       callback=callback):
            threading.Event().wait()

# ─── Main GUI ───────────────────────────────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Input vs. Enhanced: Waveform & Spectrogram")

        # Create four views
        self.raw_wave = pg.PlotWidget(title="Raw Waveform")
        self.raw_wave.setYRange(-1, 1)
        self.raw_spec = pg.ImageView()
        self.raw_spec.ui.roiBtn.hide(); self.raw_spec.ui.menuBtn.hide()

        self.enh_wave = pg.PlotWidget(title="Enhanced Waveform")
        self.enh_wave.setYRange(-1, 1)
        self.enh_spec = pg.ImageView()
        self.enh_spec.ui.roiBtn.hide(); self.enh_spec.ui.menuBtn.hide()

        # Layout: two columns
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.raw_wave, 0, 0)
        layout.addWidget(self.enh_wave, 0, 1)
        layout.addWidget(self.raw_spec, 1, 0)
        layout.addWidget(self.enh_spec, 1, 1)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Worker thread
        self.worker = AudioWorker()
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)

        # Connect signals to update functions
        self.worker.raw_wave_ready.connect(self.update_raw_wave)
        self.worker.raw_spec_ready.connect(self.update_raw_spec)
        self.worker.enh_wave_ready.connect(self.update_enh_wave)
        self.worker.enh_spec_ready.connect(self.update_enh_spec)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
        mute_sc = QShortcut(QKeySequence("M"), self)
        mute_sc.activated.connect(self.toggle_mute)

    def update_raw_wave(self, data):
        self.raw_wave.plot(data, clear=True)

    def update_raw_spec(self, spec):
        self.raw_spec.setImage(spec, autoLevels=True)

    def update_enh_wave(self, data):
        self.enh_wave.plot(data, clear=True)

    def update_enh_spec(self, spec):
        self.enh_spec.setImage(spec, autoLevels=True)
        
    def toggle_mute(self):
        # flip the flag on the worker and give visual feedback
        self.worker.muted = not self.worker.muted
        status = "MUTED" if self.worker.muted else "UNMUTED"
        print(f"Microphone is now {status}")

# ─── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
