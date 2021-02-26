import glob
import os

import numpy as np
import torch
import torchaudio


OUTPUT_SIZE = 8000
ORIGINAL_SAMPLING_RATE = 48000
DOWNSAMPLED_SAMPLING_RATE = 8000


class AudioMNISTDataset(torch.utils.data.Dataset):
    """Dataset object for the AudioMNIST data set."""
    def __init__(self, root_dir, transform=None, verbose=False):
        self.root_dir = root_dir
        self.audio_list = glob.glob(f"{root_dir}/*/*.wav")
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_fn = self.audio_list[idx]
        if self.verbose:
            print(f"Loading audio file {audio_fn}")
        waveform, sample_rate = torchaudio.load(audio_fn)
        if self.transform:
            waveform = self.transform(waveform)
        sample = {
            'input': waveform,
            'digit': int(os.path.basename(audio_fn).split("_")[0])
        }
        return sample


class PreprocessRaw(object):
    """Transform audio waveform of given shape."""
    def __init__(self, size_out=OUTPUT_SIZE, orig_freq=ORIGINAL_SAMPLING_RATE,
                 new_freq=DOWNSAMPLED_SAMPLING_RATE):
        self.size_out = size_out
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform):
        transformed_waveform = _ZeroPadWaveform(self.size_out)(
            _ResampleWaveform(self.orig_freq, self.new_freq)(waveform)
        )
        return transformed_waveform


class _ResampleWaveform(object):
    """Resample signal frequency."""
    def __init__(self, orig_freq, new_freq):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform):
        return self._resample_waveform(waveform)

    def _resample_waveform(self, waveform):
        resampled_waveform = torchaudio.transforms.Resample(
            orig_freq=self.orig_freq,
            new_freq=self.new_freq,
        )(waveform)
        return resampled_waveform


class _ZeroPadWaveform(object):
    """Apply zero-padding to waveform.

    Return a zero-padded waveform of desired output size. The waveform is
    positioned randomly.
    """
    def __init__(self, size_out):
        self.size_out = size_out

    def __call__(self, waveform):
        return self._zero_pad_waveform(waveform)

    def _zero_pad_waveform(self, waveform):
        padding_total = self.size_out - waveform.shape[-1]
        padding_left = np.random.randint(padding_total + 1)
        padding_right = padding_total - padding_left
        padded_waveform = torch.nn.ConstantPad1d(
            (padding_left, padding_right),
            0
        )(waveform)
        return padded_waveform