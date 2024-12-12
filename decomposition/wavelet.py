import numpy as np
import pywt
import torch

def decompose_wavelet(input:np.array, wavelet:str, levels:int):
    # Input shape: (C, N, L)
    all_out = []
    for channel in input:
        out_channel = []
        for sample in channel:
            out_channel.append(_decompose(sample, wavelet, levels))
        all_out.append(out_channel)

    # all_out shape: (C,N,n_decomp,L)
    # Should be: (n_decomp,C,N,L)

    # Reshape ragged
    out = np.empty((len(all_out[0][0]),len(input),len(input[0])),dtype=object)
    out.fill([])
    out = out.tolist()
    for i, c in enumerate(all_out):
        for j, s in enumerate(c):
            for k, d in enumerate(s):
                out[k][i][j] = d
    # Reshape
    # out = np.array(out).transpose(2,0,1,3)
    for subseries in out:
        yield subseries

def _decompose(signal:np.array, wavelet:str, levels:int):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    components = []
    zeros = [np.zeros_like(c, dtype=float) for c in coeffs]
    for i in range(len(coeffs)):
        zeros[i] = coeffs[i]
        components.append(pywt.waverec(zeros, wavelet)) # Partial wavelet reconstruction
        zeros[i] = np.zeros_like(zeros[i])
    return np.array(components)

def compose_wavelet(inputs):
    return torch.sum(inputs, axis=0)