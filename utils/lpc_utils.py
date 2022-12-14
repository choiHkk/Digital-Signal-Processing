"""https://github.com/revsic/torch-nansy"""

import torch.nn.functional as F
import torch



def reconstruct_fft(fft: torch.Tensor, 
                    n_code_channels: int, 
                    win_length: int, 
                    epsilon: float = 1e-6, 
                    return_complex: bool = True):
    envelope = calculate_envelope(
        fft=fft, 
        n_code_channels=n_code_channels, 
        win_length=win_length, 
        epsilon=epsilon
    )
    source = calculate_source(
        fft=fft, 
        envelope=envelope, 
        epsilon=epsilon
    )
    if not return_complex:
        source = torch.real(source)**2 + torch.imag(source)**2
        source = torch.sqrt(source + epsilon)
    fft = source * envelope
    return fft, source, envelope


def calculate_source(fft: torch.Tensor, envelope: torch.Tensor, epsilon: float = 1e-6):
    source = fft / (envelope + epsilon)
    return source


def calculate_envelope(fft: torch.Tensor, n_code_channels: int, win_length: int, epsilon: float = 1e-6):
    corrcoef = calculate_autocorration_coefficient(fft=fft, n_code_channels=n_code_channels, epsilon=epsilon)
    code = solve_toeplitz(corrcoef=corrcoef)
    fft_code = torch.fft.rfft(-F.pad(code, [1, 0], value=1.), win_length, dim=2)
    fft_code = torch.abs(fft_code)
    fft_code[(fft_code - epsilon) < 0] = 1.0
    envelope = fft_code ** -1
    return envelope.transpose(1,2)


def calculate_autocorration_coefficient(fft: torch.Tensor, n_code_channels: int, epsilon: float = 1e-6):
    fft_norm = fft / torch.unsqueeze(torch.clamp(torch.mean(torch.abs(fft), dim=1), min=epsilon), dim=1)
    corrcoef = torch.fft.irfft(torch.square(torch.abs(fft_norm)), dim=1)
    corrcoef = corrcoef[:,:n_code_channels+1,:].transpose(1,2)
    return corrcoef


def solve_toeplitz(corrcoef: torch.Tensor) -> torch.Tensor:
    """Solve the toeplitz matrix.
    Args:
        corrcoef: [torch.float32; [..., num_code + 1]], auto-correlation.
    Returns:
        [torch.float32; [..., num_code]], solutions.
    """
    ## solve the first row
    # [..., 2]
    solutions = F.pad(
        (-corrcoef[..., 1] / corrcoef[..., 0].clamp_min(1e-7))[..., None],
        [1, 0], value=1.)
    # [...]
    extra = corrcoef[..., 0] + corrcoef[..., 1] * solutions[..., 1]

    ## solve residuals
    num_code = corrcoef.shape[-1] - 1
    for k in range(1, num_code):
        # [...]z
        lambda_value = (
                -solutions[..., :k + 1]
                * torch.flip(corrcoef[..., 1:k + 2], dims=[-1])
            ).sum(dim=-1) / extra.clamp_min(1e-7)
        # [..., k + 2]
        aug = F.pad(solutions, [0, 1])
        # [..., k + 2]
        solutions = aug + lambda_value[..., None] * torch.flip(aug, dims=[-1])
        # [...]
        extra = (1. - lambda_value ** 2) * extra
    # [..., num_code]
    return solutions[..., 1:]
