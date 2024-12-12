import numpy as np
import torch

def decompose_fourier_topk(input:np.array, k:int):
    # Input shape: (C, N, L)
    all_out = []
    for channel in input:
        out_channel = []
        for sample in channel:
            out_channel.append(_decompose(sample, k))
        all_out.append(out_channel)

    # all_out shape: (C,N,n_decomp,L)
    # Should be: (n_decomp,C,N,L)
    
    # Reshape ragged
    out = np.empty((2,len(input),len(input[0])),dtype=object)
    out.fill([])
    out = out.tolist()
    for i, c in enumerate(all_out):
        for j, s in enumerate(c):
            for k, d in enumerate(s):
                out[k][i][j] = d
    # Seasonality, Residual
    return out[0], out[1]

def _decompose(signal:np.ndarray, k:int):
    coeffs = np.fft.rfft(signal)
    freq = abs(coeffs)
    freq[0] = 0
    top_k_freq = np.partition(freq, -k)[-k]
    coeffs[freq < top_k_freq] = 0
    seasonality = np.fft.irfft(coeffs,len(signal))
    residual = signal - seasonality
    return np.array([seasonality, residual])

def compose_fourier_topk(inputs):
    return torch.sum(inputs, axis=0)