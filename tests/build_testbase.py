import os
from pathlib import Path

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 256
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 5e-4
NUM_CLASSES = 37

ROOT = "./speech_commands_data"
os.makedirs(ROOT, exist_ok=True)
test_set = SPEECHCOMMANDS(ROOT, download=True, subset="testing")


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
        # Ensure waveform is on the correct device and has the right shape (channels, samples)
        if len(waveform.shape) == 1:  # Mono audio
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)

        # Convert to Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to log scale (Decibels)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Remove the channel dimension if it's 1 (e.g., (1, n_mels, frames) -> (n_mels, frames))
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0)
        return mel_spec


class CachedMelDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = sorted(os.listdir(cache_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.cache_dir, self.files[idx]))
        return data["mel"], data["label"]


def preprocess_and_cache(
    dataset,
    split_name,
    audio_processor,
    label_to_idx,
    sample_rate,
    cache_root="mel_cache",
):
    split_dir = os.path.join(cache_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    target_length = sample_rate  # 1 second

    for idx in tqdm(range(len(dataset)), desc=f"Caching {split_name}"):
        cache_path = os.path.join(split_dir, f"{idx:06d}.pt")
        if os.path.exists(cache_path):
            continue  # skip if already cached

        waveform, _, label, *_ = dataset[idx]

        # Pad / trim
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = torch.zeros(1, target_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)

        # Feature extraction
        mel = audio_processor(waveform).cpu()

        torch.save(
            {
                "mel": mel,
                "label": label_to_idx[label],
            },
            cache_path,
        )


audio_processor = AudioProcessor(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    device=DEVICE,

)
all_labels = sorted(list({Path(p).parent.name for p in test_set._walker}))

# COMMON_COMMANDS should represent the full set of labels for v0.02
# Start with all_labels, then add 'unknown' and 'silence' if not already present
COMMON_COMMANDS = all_labels.copy()
if "unknown" not in COMMON_COMMANDS:
    COMMON_COMMANDS.append("unknown")
if "silence" not in COMMON_COMMANDS:
    COMMON_COMMANDS.append("silence")

labels = COMMON_COMMANDS
label_to_idx = {l: i for i, l in enumerate(labels)}
NUM_CLASSES = len(labels)
print(NUM_CLASSES)

preprocess_and_cache(
    test_set,
    split_name="test",
    audio_processor=audio_processor,
    label_to_idx=label_to_idx,
    sample_rate=SAMPLE_RATE,
)


ds = CachedMelDataset("mel_cache/test")

os.makedirs("mel_bin", exist_ok=True)
n = 100
for i in range(n):
    mel, label = ds[i]

    mel = mel.cpu().contiguous().float()
    mel.numpy().tofile(f"mel_bin/mel_{i:03d}.bin")
    with open(f"mel_bin/mel_{i:03d}.txt", "w") as f:
        f.write(label)
