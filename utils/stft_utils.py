"""
https://pytorch.org/docs/stable/generated/torch.stft.html
https://pytorch.org/docs/stable/generated/torch.istft.html
https://github.com/jaywalnut310/vits/blob/main/mel_processing.py
"""

from typing import Optional
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn


functions = {
    "mel_basis": {}, 
    "hann_window": {}
}


def linear_spectrogram(y: torch.Tensor, 
                       n_fft, 
                       hop_length, 
                       win_length: Optional[int] = None, 
                       center: bool = False, 
                       pad_mode: str = 'reflect', 
                       normalized: bool = False, 
                       onesided: bool = True, 
                       return_complex: bool = True, 
                       spectral_normalize: bool = False, 
                       epsilon=1e-6):
    if win_length is None:
        win_length = n_fft
        
    y = stft(
        y=y, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        center=center, 
        pad_mode=pad_mode, 
        normalized=normalized, 
        onesided=onesided, 
        return_complex=return_complex
    )
    if return_complex:
        y = torch.real(y)**2 + torch.imag(y)**2
    else:
        y = y.pow(2).sum(-1)
    y = torch.sqrt(y + epsilon)
    if spectral_normalize:
        y = normalize(y=y, epsilon=epsilon)
    return y


def mel_spectrogram(y: torch.Tensor, 
                    sampling_rate: int, 
                    n_mel_channels: int, 
                    n_fft: int, 
                    hop_length: int, 
                    win_length: Optional[int] = None, 
                    mel_fmin: Optional[int] = None, 
                    mel_fmax: Optional[int] = None, 
                    center: bool = False, 
                    pad_mode: str = 'reflect', 
                    normalized: bool = False, 
                    onesided: bool = True, 
                    return_complex: bool = True, 
                    spectral_normalize: bool = False, 
                    epsilon=1e-6):
    if mel_fmin is None:
        mel_fmin = 0
    if mel_fmax is None:
        mel_fmax = sampling_rate // 2
    if win_length is None:
        win_length = n_fft
        
    y = linear_spectrogram(
        y=y, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        center=center, 
        pad_mode=pad_mode, 
        normalized=normalized, 
        onesided=onesided, 
        return_complex=return_complex, 
        epsilon=epsilon, 
        spectral_normalize=False
    )
    y = linear_to_mel(
        y=y, 
        sampling_rate=sampling_rate, 
        n_fft=n_fft, 
        n_mel_channels=n_mel_channels, 
        mel_fmin=mel_fmin, 
        mel_fmax=mel_fmax, 
        epsilon=epsilon, 
        spectral_normalize=False
    )
    if spectral_normalize:
        y = normalize(y=y, epsilon=epsilon)
    return y


def linear_to_mel(y: torch.Tensor, 
                  sampling_rate: int, 
                  n_fft: int, 
                  n_mel_channels: int, 
                  mel_fmin: Optional[int] = None, 
                  mel_fmax: Optional[int] = None, 
                  epsilon: float = 1e-6, 
                  spectral_normalize: bool = False):
    global functions
    if mel_fmin is None:
        mel_fmin = 0
    if mel_fmax is None:
        mel_fmax = sampling_rate // 2
        
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(mel_fmin) + '_' + dtype_device
    if fmax_dtype_device not in functions["mel_basis"]:
        mel_fn = librosa_mel_fn(
            sr=sampling_rate, 
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax
        )
        functions["mel_basis"][fmax_dtype_device] = torch.from_numpy(mel_fn).to(dtype=y.dtype, device=y.device)
    y = torch.matmul(functions["mel_basis"][fmax_dtype_device], y)
    if spectral_normalize:
        y = normalize(y=y, epsilon=epsilon)
    return y


def reconstruct_wav(y: torch.Tensor, 
                    n_fft: int, 
                    hop_length: int, 
                    win_length: int):
    fft = stft(
        y=y, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        return_complex=False
    )
    y = istft(
        y=fft, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        return_complex=False
    )
    return y


def stft(y: torch.Tensor, 
         n_fft, 
         hop_length, 
         win_length: Optional[int] = None, 
         center: bool = False, 
         pad_mode: str = 'reflect', 
         normalized: bool = False, 
         onesided: bool = True, 
         return_complex: bool = False):
    
    global functions
    if win_length is None:
        win_length = n_fft
    
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_length) + '_' + dtype_device
    if wnsize_dtype_device not in functions["hann_window"]:
        functions["hann_window"][wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)
    
    y = pad(y=y, n_fft=n_fft, hop_length=hop_length, mode=pad_mode)
    fft = torch.stft(
        input=y, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=functions["hann_window"][wnsize_dtype_device], 
        center=center, 
        pad_mode=pad_mode, 
        normalized=normalized, 
        onesided=onesided, 
        return_complex=return_complex
    )
    assert fft.size(1) == (n_fft // 2 + 1)
    return fft


def istft(y: torch.Tensor, 
          n_fft, 
          hop_length, 
          win_length: Optional[int] = None, 
          center: bool = True, 
          normalized: bool = False, 
          onesided: bool = True, 
          length: Optional[int] = None, 
          return_complex: bool = False):
    
    global functions
    if win_length is None:
        win_length = n_fft
    
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_length) + '_' + dtype_device
    if wnsize_dtype_device not in functions["hann_window"]:
        functions["hann_window"][wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)
    
    y = torch.istft(
        input=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=functions["hann_window"][wnsize_dtype_device], 
        center=center, 
        normalized=normalized, 
        onesided=onesided, 
        length=length, 
        return_complex=return_complex
    )
    return y


def normalize(y: torch.Tensor, epsilon: float = 1e-6):
    return torch.log(torch.clamp(input=y, min=epsilon))


def denormalize(y: torch.Tensor):
    return torch.exp(y)


def pad(y: torch.Tensor, n_fft: int, hop_length: int, mode: str = "reflect"):
    pad = (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2))
    y = F.pad(input=y.unsqueeze(1), pad=pad, mode=mode).squeeze(1)
    return y

