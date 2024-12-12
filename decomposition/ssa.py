import numpy as np
import torch
from pyts.decomposition import SingularSpectrumAnalysis

def decompose_ssa(input:np.array, n_subseries:int, window_size:int):
    l = (window_size // n_subseries) + 1
    w = n_subseries * l
    groups = [np.arange(i*l,i*l+l) for i in range(n_subseries)]
    ssa = SingularSpectrumAnalysis(window_size=w, groups=groups)

    n_channels = len(input)
    all_out = [[] for _ in range(n_subseries)]
    for channel in input:
        if len(channel) == 1 or len(channel[0]) != len(channel[1]): # Ragged length
            for instance in channel:
                result = ssa.transform(instance[np.newaxis,:])[0]
                for r in range(n_subseries):
                    all_out[r].append(result[r])
        else:
            result = ssa.transform(channel)
            result = np.swapaxes(result, 0, 1) # Swap axes so that the subseries are at axis 0
            for r in range(n_subseries):
                all_out[r].extend(result[r])
    
    segment_length = len(all_out[0]) / n_channels
    assert int(segment_length) == segment_length
    segment_length = int(segment_length)

    # Reshape channels
    # all_out has the shape (n_subseries, N,L)
    out = [[] for _ in range(n_subseries)]
    for i, o in enumerate(all_out):
        res = []
        for j in range(n_channels):
            res.append(o[j*segment_length:(j+1)*segment_length])
        out[i] = res

    for subseries in out:
        yield subseries

def compose_ssa(inputs):
    return torch.sum(inputs,axis=0)