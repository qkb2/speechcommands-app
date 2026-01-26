import sys
import time

import numpy as np
import serial
import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256

SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 115200
SERIAL_TIMEOUT = 2.0  # seconds


class AudioProcessor:
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        device=DEVICE,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalized=True,
        ).to(self.device)

    def __call__(self, waveform):
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0)
        return mel_spec


def send_single_sample(file_path, audio_processor, ser):
    target_length = SAMPLE_RATE  # 1 second

    waveform, sr = torchaudio.load(file_path)

    # Pad / trim
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        padding = torch.zeros(1, target_length - waveform.shape[1])
        waveform = torch.cat([waveform, padding], dim=1)

    mel = audio_processor(waveform).cpu().contiguous().float()
    mel_bytes = mel.cpu().numpy().astype(np.float32).tobytes()

    ser.reset_input_buffer()
    start = time.perf_counter()

    ser.write(mel_bytes)
    ser.flush()

    # Read Pico response (one line)
    response = ser.readline()

    end = time.perf_counter()
    inference_time = end - start

    predicted = response.decode("utf-8", errors="ignore").strip()

    print(f"Predicted: {predicted}. Time: {inference_time}")


if __name__ == "__main__":
    audio_processor = AudioProcessor()

    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, timeout=SERIAL_TIMEOUT)

    # Give Pico time to reset after USB connect
    time.sleep(2)
    ser.reset_input_buffer()

    file_name = sys.argv[1]
    send_single_sample(file_name, audio_processor, ser)
