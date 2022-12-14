import dataclasses



@dataclasses.dataclass
class Parameters(object):
    def __init__(self, 
                 sampling_rate: int = 22050, 
                 n_fft: int = 1024, 
                 hop_length: int = 256, 
                 win_length: int = 1024, 
                 n_mel_channels: int = 80,
                 n_code_channels: int = 32, 
                 spectral_normalize: bool = True, 
                 epsilon: float = 1e-6):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_code_channels = n_code_channels
        self.spectral_normalize = spectral_normalize
        self.epsilon = epsilon
