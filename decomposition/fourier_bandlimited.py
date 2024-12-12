import numpy as np
import torch

def decompose_fourier_bandlimited(input:np.array, n_bands:int):
    # Input shape: (C, N, L)
    all_out = []
    for channel in input:
        out_channel = []
        for sample in channel:
            out_channel.append(_decompose(sample, n_bands))
        all_out.append(out_channel)

    # all_out shape: (C,N,n_decomp,L)
    # Should be: (n_decomp,C,N,L)
    
    # Reshape ragged
    out = np.empty((n_bands,len(input),len(input[0])),dtype=object)
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

def _decompose(signal:np.ndarray, n_bands:int):
    coeffs = np.fft.fft(signal)

    components = []
    band_mask = np.zeros_like(signal, dtype=complex)

    step_size = int(np.ceil(len(signal) / n_bands))
    starts = np.arange(n_bands) * step_size
    ends = (np.arange(n_bands) + 1) * step_size
    for s,e in zip(starts, ends):
        band_mask[s:e] = coeffs[s:e]
        
        component = np.fft.ifft(band_mask)
        components.append(np.real(component))

        band_mask[s:e] = np.zeros_like(band_mask[s:e])

    return np.array(components)

def compose_fourier_bandlimited(inputs):
    return torch.sum(inputs, axis=0)