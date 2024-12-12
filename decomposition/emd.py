import emd
import numpy as np
import torch 

def decompose_emd(input:np.array, n_subseries:int):
    emd.logger.set_up(level="ERROR")
    n_channels = len(input)
    all_out = [[] for _ in range(n_subseries)]
    for channel in input:
        for instance in channel:
            imfs = emd.sift.sift(instance, max_imfs=n_subseries-1, sift_thresh=None, energy_thresh=None, rilling_thresh=None).T
            if len(imfs) < n_subseries:
                imfs = np.concatenate([imfs, np.zeros((n_subseries-len(imfs),imfs.shape[1]))])
            for r in range(n_subseries):
                all_out[r].append(imfs[r])
    
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

def compose_emd(inputs):
    return torch.sum(inputs,axis=0)