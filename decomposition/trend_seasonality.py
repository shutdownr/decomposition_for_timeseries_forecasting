import numpy as np
import torch
from decomposition.moving_average import _decompose as _decompose_moving_average

def decompose_trend_seasonality(input:np.array, kernel_size:int, period:int, pad_mean_size=5):
    assert kernel_size % 2 == 1
    if period == 1: # Fix, as period = 1 doesn't work.
        return input, np.zeros_like(input), np.zeros_like(input)
    # Input shape: (C, N, L)
    all_out = []
    for channel in input:
        out_channel = []
        for sample in channel:
            out_channel.append(_decompose(sample, kernel_size, period, pad_mean_size))
        all_out.append(out_channel)

    # all_out shape: (C,N,n_decomp,L)
    # Should be: (n_decomp,C,N,L)
    
    # Reshape ragged
    out = np.empty((3,len(input),len(input[0])),dtype=object)
    out.fill([])
    out = out.tolist()
    for i, c in enumerate(all_out):
        for j, s in enumerate(c):
            for k, d in enumerate(s):
                out[k][i][j] = d
    # Trend, Seasonality, Residual
    return out[0], out[1], out[2]

def _decompose(signal:np.ndarray, kernel_size:int, period:int, pad_mean_size:int):
    trend, res = _decompose_moving_average(signal, kernel_size=kernel_size, pad_mean_size=pad_mean_size)
    pad_len = len(signal) % period
    front = np.full(pad_len, np.mean(signal[:pad_mean_size]))
    end = np.full(pad_len, np.mean(signal[-pad_mean_size:]))
    signal_padded = np.concatenate([front, signal, end],axis=-1)

    out_seasonal = np.zeros_like(trend)
    out_residual = np.zeros_like(trend)
    for i in range(period):
        seasonal = np.mean(signal_padded[i::period],keepdims=True)
        residual = res[i::period] - seasonal
        out_seasonal[i::period] = seasonal
        out_residual[i::period] = residual
    return trend, out_seasonal, out_residual

def compose_trend_seasonality(inputs):
    return torch.sum(inputs,axis=0)