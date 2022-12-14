import torch.nn.functional as F
import torch



def interpolate_complex(input: torch.Tensor, scale_factor: float = 1.0, mode: str = "linear"):
    global functions
    magnitude = F.interpolate(torch.abs(input), scale_factor=scale_factor, mode=mode)
    angle = F.interpolate(torch.angle(input), scale_factor=scale_factor, mode=mode)
    return torch.polar(magnitude, angle)


def interpolate_complex2(input: torch.Tensor, scale_factor: float = 1.0, mode: str = "linear"):
    global functions
    real = F.interpolate(torch.real(input), scale_factor=scale_factor, mode=mode)
    imag = F.interpolate(torch.imag(input), scale_factor=scale_factor, mode=mode)
    return torch.complex(real, imag)


functions = {
    torch.float32: F.interpolate, 
    torch.complex64: interpolate_complex2
}


def interpolate(fft: torch.Tensor, scale_factor: float = 1.0, mode: str = "linear", epsilon: float = 1e-6):
    global functions
    fft = fft.transpose(1,2)
    H = fft.size(2)
    fft = functions[fft.dtype](input=fft, scale_factor=scale_factor, mode=mode)[...,:H]
    fft = F.pad(fft, (0, H - fft.size(2))).transpose(1,2)
    return fft
