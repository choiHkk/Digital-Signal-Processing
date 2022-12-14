"""https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html"""

from typing import Optional, Any
from scipy.io import wavfile
from scipy import signal
import IPython.display as ipd
import numpy as np


"""
32-bit floating-point: -1.0 ~ +1.0 [float32]
32-bit integer PCM: -2147483648 ~ +2147483647 [int32]
24-bit integer PCM: -2147483648 ~ +2147483392 [int32]
16-bit integer PCM: -32768 ~ +32767 [int16]
8-bit integer PCM: 0 ~ 255 [uint8]
"""


MAX_WAV_VALUES = {
    "int16": 2**16/2, # 32768.
    "float32": 1., 
    "uint8": 2**8/2,   # 128.
    "int32":  {
        "1": 2**24/2, # 8388608. 
        "2": 2**32/2  # 2147483648.
    }
}


def read_wav(filename: str, target_sampling_rate: Optional[int] = None):
    sampling_rate, y = wavfile.read(filename=filename)
    y = validate_wav(y=y)
    if target_sampling_rate is not None:
        sampling_rate, y = resample_wav(
            y=y, 
            source_sampling_rate=sampling_rate, 
            target_sampling_rate=target_sampling_rate
        )
    return sampling_rate, y


def write_wav(filename: str, target_sampling_rate: int, y: np.ndarray):
    wavfile.write(filename=filename, rate=target_sampling_rate, data=y)


def resample_wav(y: np.ndarray, source_sampling_rate: int, target_sampling_rate: int):
    length = round(y.shape[-1] * target_sampling_rate / source_sampling_rate)
    y = signal.resample(x=y, num=length)
    return target_sampling_rate, y


def validate_wav(y: np.ndarray):
    dtype = y.dtype
    if dtype == "int16":
        max_wav_value = MAX_WAV_VALUES["int16"]
    elif dtype == "float32":
        max_wav_value = MAX_WAV_VALUES["float32"]
    elif dtype == "uint8":
        max_wav_value = MAX_WAV_VALUES["uint8"] # 128
        y = y - max_wav_value
    elif dtype == "int32":
        if np.max(np.abs(y)) <= 2**24/2:
            max_wav_value = MAX_WAV_VALUES["int32"]["1"] # 8388608.
        else:
            max_wav_value = MAX_WAV_VALUES["int32"]["2"] # 2147483648.
    else:
        raise ValueError(f"{dtype}: {np.min(y)} ~ {np.max(y)}")
    if len(y.shape) > 1:
        y = np.mean(y, axis=-1)
    y = y / max_wav_value
    return y


def visualize_wav(y: np.ndarray, sampling_rate: int, autoplay: bool = False, normalize: bool = False):
    display(ipd.Audio(data=y, rate=sampling_rate, autoplay=autoplay, normalize=normalize))
