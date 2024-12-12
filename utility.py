import random

import numpy as np
import torch

from forecasting_dataset import ForecastingDataset

def set_random_seed(seed):
    random.seed(int(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Transforms all decomposed components to channel-like features, returns the dataset as (N, c*C, L)
def components_to_features(forecasting_dataset:ForecastingDataset, variable_name:str):
    component_names = forecasting_dataset.decomposition_variables
    # Shape: (c,N,C,L), where c is the number of components
    components_stacked = np.stack([forecasting_dataset.get_attr_flat(variable_name+"_"+comp, order="NCL") for comp in component_names], axis=0)
    # (N,c,C,L)
    components_stacked = components_stacked.transpose(1,0,2,3)
    # (N, c*C, L)
    components_stacked = components_stacked.reshape(components_stacked.shape[0],-1,components_stacked.shape[-1])
    return components_stacked

def format_algorithm_name(algorithm, long=True):
    if algorithm == "DLinear":
        return "DLinear"
    elif algorithm == "GenericLinear":
        return "\\texttt{G-Linear}"
    elif algorithm == "NBeats":
        return "N-BEATS"
    elif algorithm == "GenericNBeats":
        return "\\texttt{G-BEATS}"
    elif algorithm == "GRU":
        return "GRU"
    elif algorithm == "GenericGRU":
        return "\\texttt{G-GRU}"
    elif algorithm == "NonStationaryTransformer":
        return "Non-Stationary Transformer" if long else "NST"
    elif algorithm == "TimeMixer":
        return "TimeMixer" if long else "T.Mixer"
    elif algorithm == "GenericTimeMixer":
        return "\\texttt{G-TimeMixer}"
    elif algorithm == "iTransformer":
        return "iTransformer" if long else "iTransf."
    else:
        return "Unknown algorithm"

def format_decomp_name(decomp, long=True):
    if decomp == "none":
        return "None"
    elif decomp == "moving_avg":
        return "Moving-average" if long else "MA"
    elif decomp == "trend_seasonality":
        return "Trend-seasonality" if long else "TS"
    elif decomp == "STL":
        return "STL"
    elif decomp == "SSA":
        return "SSA"
    elif decomp == "fourier_bandlimited":
        return "Fourier (Bandlimited)" if long else "F(BL)"
    elif decomp == "fourier_topk":
        return "Fourier (Top-k)" if long else "F(k)"
    elif decomp == "wavelet":
        return "Wavelet" if long else "Wav"
    elif decomp == "EMD":
        return "EMD"
    else:
        return "Unknown decomp"
    
def format_dataset_name(dataset, long=True):
    if dataset == "m4_h":
        return "M4 Hourly" if long else "M4(H)"
    elif dataset == "weather_uts":
        return "Weather"
    elif dataset == "transactions":
        return "Transactions" if long else "Trans."
    elif dataset == "cif":
        return "CIF"
    elif dataset == "rain":
        return "Rain"
    elif dataset == "m4_y":
        return "M4 Yearly" if long else "M4(Y)"
    elif dataset == "covid":
        return "Covid"
    elif dataset == "ett_h1":
        return "ETTh1"
    elif dataset == "exchange_rate":
        return "Exchange Rate" if long else "Exch."
    elif dataset == "illness":
        return "Illness"
    elif dataset == "walmart":
        return "Walmart"
    # For plotting purposes
    elif dataset in ["Average", "Mean", "Mean (S)", "Mean (L)"]:
        return dataset
    else:
        return "Unknown dataset"
