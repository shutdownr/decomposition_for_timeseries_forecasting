import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse 

class ssa_decomposition(nn.Module):
    """
    Singular Spectrum Analysis decomposition block
    """
    def __init__(self, window_size, n_components=4):
        super(ssa_decomposition, self).__init__()
        self.window_size = window_size
        self.n_components = n_components

    def forward(self, x):   
        B,C,N = x.shape
        L = self.window_size
        K = N - L + 1

        new_shape = (B,C,K,L)
        new_strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(2))

        trajectory_matrix = x.as_strided(size=new_shape, stride=new_strides)

        # Perform SVD
        U, Sigma, VT = torch.linalg.svd(trajectory_matrix)
        # Number of components
        d = self.n_components
        # trajectory_matrix_rank = torch.linalg.matrix_rank(trajectory_matrix[0][0])
        # d = trajectory_matrix_rank
        components = []
        # Reconstruct the elementary matrices without storing them
        for i in range(d):
            comp = torch.zeros((B, C, N))
            X_elem = Sigma[:,:,i:i+1].unsqueeze(-1) * torch.einsum('ijk,ijl->ijlk', U[:,:,:,i], VT[:,:,i,:])
            X_rev = X_elem.flip(-2) # Flip for diagonals
            for j, n in zip(range(-X_rev.shape[-2]+1, X_rev.shape[-1]), range(N)):
                comp[:,:,n] = X_rev.diagonal(j,dim1=-2,dim2=-1).mean(dim=-1)
            components.append(comp)
        return torch.stack(components)

class wavelet_decomposition(nn.Module):
    """
    Wavelet-based decomposition block
    """

    def __init__(self, n_levels=2, wavelet="db4"):
        super(wavelet_decomposition, self).__init__()

        self.n_levels = n_levels
        self.wavelet = wavelet

        self.idwt = DWT1DInverse(wave=self.wavelet)
        self.dwt = DWT1DForward(wave=self.wavelet, J=self.n_levels)

        self.n_components = n_levels + 1 # Approximation + n_levels details

    def forward(self, x):
        yl, yh = self.dwt(x)

        mask_yh = [torch.zeros_like(c) for c in yh]

        out = [self.idwt((yl, mask_yh))] # Approximation
        zeros_yl = torch.zeros_like(yl)        

        for i in range(len(mask_yh)):
            masked_yh = [mask_yh[i] * yh[i] if idx == i else torch.zeros_like(yh[idx]) for idx in range(len(yh))]
            out.append(self.idwt((zeros_yl, masked_yh))) # Details
        return torch.stack(out)

class fourier_bandlimited_decomposition(nn.Module):
    """
    Fourier-based decomposition block (N bandlimited signals)
    """

    def __init__(self, n_bands=4):
        super(fourier_bandlimited_decomposition, self).__init__()
        self.n_bands = n_bands
        self.n_components = n_bands

    def forward(self, x):
        coeffs = torch.fft.fft(x)

        step_size = coeffs.shape[-1] / self.n_bands
        if step_size % 1 > 0:
            step_size = step_size // 1
        step_size = int(step_size)

        components = []
        for i in range(self.n_bands):
            coeff_mask = torch.zeros_like(coeffs)
            coeff_mask[...,i*step_size:(i+1)*step_size] = 1
            components.append(torch.fft.ifft(coeffs*coeff_mask).real)
        return torch.stack(components)

class fourier_topk_decomposition(nn.Module):
    """
    Fourier-based decomposition block (Top k frequencies)
    """

    def __init__(self, top_k=5):
        super(fourier_topk_decomposition, self).__init__()
        self.top_k = top_k
        self.n_components = 2

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_residual = x - x_season
        return x_residual, x_season

class trend_seasonality_decomposition(nn.Module):
    """
    Trend-seasonality-based decomposition block
    """

    def __init__(self, kernel_size, periodicity):
        super(trend_seasonality_decomposition, self).__init__()
        self.moving_avg = moving_average_decomposition(kernel_size, stride=1, padding_mode="median")
        self.periodicity = periodicity
        self.n_components = 3

    def forward(self, x):
        res, out_trend = self.moving_avg(x)

        out_seasonal = torch.zeros_like(res)
        out_residual = torch.zeros_like(res)
        pad_len = res.shape[-1] % self.periodicity
        if pad_len > 0:
            x_padded = torch.cat((res,res[:,:,-self.periodicity:-pad_len]),dim=-1)
        else:
            x_padded = res

        for i in range(self.periodicity):
            seasonal = torch.mean(x_padded[:,:,i::self.periodicity],dim=2,keepdim=True)
            residual = res[:,:,i::self.periodicity] - seasonal
            out_seasonal[:,:,i::self.periodicity] = seasonal
            out_residual[:,:,i::self.periodicity] = residual

        return out_residual, out_trend, out_seasonal

class moving_average_decomposition(nn.Module):
    """
    Moving average-based decomposition block
    """

    def __init__(self, kernel_size, stride=1, padding_mode="last"):
        super(moving_average_decomposition, self).__init__()
        self.kernel_size = kernel_size
        self.padding_mode=padding_mode
        self.n_components = 2

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def pad(self, x):
        if self.padding_mode == "last":
            front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
            end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        elif self.padding_mode == "mean":
            pad_length = 5
            front = torch.mean(x[:, :, :pad_length],dim=2,keepdim=True).repeat(1, 1, (self.kernel_size - 1) // 2)
            end = torch.mean(x[:, :, -pad_length:],dim=2,keepdim=True).repeat(1, 1, (self.kernel_size - 1) // 2)
        elif self.padding_mode == "median":
            pad_length = 5
            front = torch.median(x[:, :, :pad_length],dim=2,keepdim=True).values.repeat(1, 1, (self.kernel_size - 1) // 2)
            end = torch.median(x[:, :, -pad_length:],dim=2,keepdim=True).values.repeat(1, 1, (self.kernel_size - 1) // 2)

        return torch.cat([front, x, end], dim=2)

    def forward(self, x):
        # padding on the both ends of time series
        x_padded = self.pad(x)
        moving_mean = self.avg(x_padded)
        res = x - moving_mean
        return res, moving_mean

class no_decomposition(nn.Module):
    """
    Empty block for testing purposes
    """

    def __init__(self):
        super(no_decomposition, self).__init__()
        self.n_components = 1

    def forward(self, x):
        return x.unsqueeze(0)