import numpy as np
import torch

def decompose_moving_average(input:np.array, kernel_size:int, pad_mean_size=5):
    assert kernel_size % 2 == 1
    # Input shape: (C, N, L)
    all_out = []
    for channel in input:
        out_channel = []
        for sample in channel:
            out_channel.append(_decompose(sample, kernel_size, pad_mean_size))
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

def _decompose(signal:np.ndarray, kernel_size:int, pad_mean_size:int):
    pad_size = (kernel_size - 1) // 2
    front = np.full(pad_size, np.mean(signal[:pad_mean_size]))
    end = np.full(pad_size, np.mean(signal[-pad_mean_size:]))
    signal_padded = np.concatenate([front, signal, end],axis=-1)[np.newaxis,:]
    trend = torch.nn.functional.avg_pool1d(torch.tensor(signal_padded), kernel_size=kernel_size, stride=1, padding=0).numpy()[0]
    residual = signal - trend
    return trend, residual

def compose_moving_average(inputs):
    return torch.sum(inputs,axis=0)
