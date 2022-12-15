from nnAudio import Spectrogram
import torch.nn as nn
import torch



class CQT(nn.Module):
    def __init__(self, 
                 sampling_rate: int, 
                 hop_length: int = 256, 
                 fmin: float = 32.7, 
                 bins_per_octave: int = 24, 
                 n_bins: int = 191):
        super(CQT, self).__init__()
        self.cqt_layer = Spectrogram.CQT2010v2(
            sr=sampling_rate, 
            hop_length=hop_length, 
            fmin=fmin, 
            fmax=sampling_rate//2, 
            bins_per_octave=bins_per_octave, 
            n_bins=n_bins, 
            verbose=False
        )
        
    def forward(self, y, normalize=True):
        cqts = self.cqt_layer(y)
        if normalize:
            cqts = torch.log(cqts + 1e-2)
        return cqts


class AugmentationCQT(nn.Module):
    def __init__(self, 
                 sampling_rate: int, 
                 hop_length: int = 256, 
                 fmin: float = 32.7, 
                 bins_per_octave: int = 24, 
                 n_bins: int = 191):
        super(AugmentationCQT, self).__init__()
        self.cqt_layer = Spectrogram.CQT2010v2(
            sr=sampling_rate, 
            hop_length=hop_length, 
            fmin=fmin, 
            fmax=sampling_rate//2, 
            bins_per_octave=bins_per_octave, 
            n_bins=n_bins, 
            verbose=False
        )
        
    def forward(self, y, p=15, normalize=True):
        B = y.size(0)
        R = torch.randint(-12,12,(B,))
        cqts = self.cqt_layer(y)
        if normalize:
            cqts = torch.log(cqts + 1e-2)
        cqts_sliced = cqts.clone()[:,p+1:-p,:]
        cqts_sliced_shifted = []
        for cqt, r in zip(cqts, R):
            r = r.item()
            cqts_sliced_shifted.append(cqt.clone()[p+1+r:-p+r,:])
        cqts_sliced_shifted = torch.stack(cqts_sliced_shifted)
        return {
            "cqt_total":cqts, 
            "cqt_sliced": cqts_sliced, 
            "cqts_sliced_shifted": cqts_sliced_shifted, 
            "r": r
        }
